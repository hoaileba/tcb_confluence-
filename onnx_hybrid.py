import torch
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
from typing import List, Dict, Tuple
from collections import defaultdict
import time

class HybridEncoder:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession("../onnx_format/model.onnx", providers=providers)
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

    def encode_batch(self, texts: List[str]) -> Tuple[List[np.ndarray], List[Dict[int, float]]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
        print(inputs["input_ids"].shape)
        print(inputs["attention_mask"].shape)
        
        output = self.session.run(None, {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]})
        last_hidden_state = output[0]
        attentions = output[2:]

        # Dense vectors (mean pooling with numpy)
        mask = np.expand_dims(inputs['attention_mask'], -1).astype(np.float32)
        token_embeddings = last_hidden_state.astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask, axis=1)
        sum_mask = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
        dense_vecs = sum_embeddings / sum_mask
        dense_vecs = dense_vecs / np.linalg.norm(dense_vecs, ord=2, axis=1, keepdims=True)

        # Sparse vectors
        sparse_vecs = []
        cls_attention = np.mean(attentions[-1], axis=1)[:, 0, :]
        
        special_ids = set(self.tokenizer.all_special_ids)
        for i in range(len(texts)):
            input_ids = inputs['input_ids'][i]
            scores = cls_attention[i]
            sparse_vec = {}
            for token_id, score in zip(input_ids, scores):
                if token_id in special_ids or score <= 0:
                    continue
                val = score * 100.0
                if token_id in sparse_vec:
                    sparse_vec[token_id] = max(sparse_vec[token_id], val)
                else:
                    sparse_vec[token_id] = val
            sparse_vecs.append(sparse_vec)

        return dense_vecs, sparse_vecs

class HybridIndex:
    def __init__(self):
        self.documents = []
        self.dense_matrix = None 
        self.inverted_index = defaultdict(dict)

    def add_batch(self, dense_vecs: List[np.ndarray], sparse_vecs: List[Dict[int, float]], payloads: List[Dict]):
        start_idx = len(self.documents)
        self.documents.extend(payloads)
        
        dense_array = np.array(dense_vecs)
        if self.dense_matrix is None:
            self.dense_matrix = dense_array
        else:
            self.dense_matrix = np.vstack([self.dense_matrix, dense_array])
            
        for doc_idx, sparse_vec in enumerate(sparse_vecs, start=start_idx):
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


if __name__ == "__main__":
    encoder = HybridEncoder()
    db = HybridIndex()
    raw_data = [
        {"id": "doc1", "text": "Kiến trúc RAG giúp AI trả lời chính xác hơn dựa trên dữ liệu riêng. Việc chunking văn bản là bước quan trọng."},
        {"id": "doc2", "text": "BM42 algorithms use attention mechanisms to create sparse vectors efficiently. It is better than TF-IDF."},
        {"id": "doc3", "text": "Món phở là đặc sản của Việt Nam, nước dùng được hầm từ xương bò trong nhiều giờ."},
        {"id": "doc4", "text": "To optimize LLM inference, we can use quantization techniques like GPTQ or AWQ on GPUs."}
    ]
    
    print("Indexing data")
    all_chunks = []
    all_payloads = []
    for item in raw_data:
        chunks = encoder.chunk_text(item["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_payloads.append({"parent_id": item["id"], "text": chunk})
    
    if all_chunks:
        start_time = time.time()
        dense_vecs, sparse_vecs = encoder.encode_batch(all_chunks)
        end_time = time.time()
        print(f"Time taken for batch encoding: {end_time - start_time} seconds")
        db.add_batch(dense_vecs, sparse_vecs, all_payloads)