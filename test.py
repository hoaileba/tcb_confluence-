from hybrid import HybridEncoder, HybridIndex
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
    for item in raw_data:
        chunks = encoder.chunk_text(item["text"])
        for chunk in chunks:
            dense, sparse = encoder.encode(chunk)
            db.add(dense, sparse, {"parent_id": item["id"], "text": chunk})
            

    def run_query(query):
        print(f"Query: '{query}'")
        q_dense, q_sparse = encoder.encode(query)
        results = db.search_hybrid(q_dense, q_sparse, top_k=2)
        
        for res in results:
            print(f"  Option Score: {res['score']:.4f} | Text: {res['payload']['text']}")

    run_query("làm sao để AI trả lời đúng hơn")
    
    run_query("thuật toán AI")
    
    run_query("món ăn từ xương bò")