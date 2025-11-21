import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
from collections import defaultdict

class HybridEncoder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded")

    def chunk_text(self, text: str, max_tokens: int = 512, overlap: int = 32) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= max_tokens:
            return [text]
        chunks = []
        stride = max_tokens - overlap
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + max_tokens]
            chunks.append(self.tokenizer.convert_tokens_to_string(chunk_tokens))
        return chunks

    def encode(self, text: str):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        token_embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        dense_vec = sum_embeddings / sum_mask
        dense_vec = torch.nn.functional.normalize(dense_vec, p=2, dim=1)

        # --- Sparse (Attention-based BM42) ---
        cls_attention = torch.mean(outputs.attentions[-1], dim=1)[:, 0, :]
        
        input_ids = inputs['input_ids'][0].cpu().numpy()
        scores = cls_attention[0].cpu().numpy()
        sparse_vec = {}
        special_ids = set(self.tokenizer.all_special_ids)
        
        for token_id, score in zip(input_ids, scores):
            if token_id in special_ids or score <= 0:
                continue
            val = score * 100.0
            if token_id in sparse_vec:
                sparse_vec[token_id] = max(sparse_vec[token_id], val)
            else:
                sparse_vec[token_id] = val
                
        return dense_vec[0].cpu().numpy(), sparse_vec

class HybridIndex:
    def __init__(self):
        self.documents = []
        self.dense_matrix = None 
        self.inverted_index = defaultdict(dict)

    def add(self, dense_vec, sparse_vec, payload):
        doc_idx = len(self.documents)
        self.documents.append(payload)
        
        if self.dense_matrix is None:
            self.dense_matrix = np.array([dense_vec])
        else:
            self.dense_matrix = np.vstack([self.dense_matrix, dense_vec])
            
        for token_id, score in sparse_vec.items():
            self.inverted_index[token_id][doc_idx] = score

    def search_dense(self, query_vec, top_k=5):
        if self.dense_matrix is None: return []
        
        scores = np.dot(query_vec, self.dense_matrix.T)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(idx, scores[idx]) for idx in top_indices]

    def search_sparse(self, query_sparse_vec, top_k=5):
        doc_scores = defaultdict(float)
        
        for token_id, query_weight in query_sparse_vec.items():
            if token_id in self.inverted_index:
                for doc_idx, doc_weight in self.inverted_index[token_id].items():
                    doc_scores[doc_idx] += query_weight * doc_weight
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_docs

    def search_hybrid(self, dense_vec, sparse_vec, top_k=5, rrf_k=60):
        dense_hits = self.search_dense(dense_vec, top_k=top_k*2)
        sparse_hits = self.search_sparse(sparse_vec, top_k=top_k*2)
        
        rrf_scores = defaultdict(float)
        
        for rank, (doc_idx, score) in enumerate(dense_hits):
            rrf_scores[doc_idx] += 1 / (rrf_k + rank + 1)
            
        for rank, (doc_idx, score) in enumerate(sparse_hits):
            rrf_scores[doc_idx] += 1 / (rrf_k + rank + 1)
            
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_idx, score in sorted_rrf:
            results.append({
                "score": score,
                "payload": self.documents[doc_idx]
            })
        return results

