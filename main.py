# main.py

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import models.database as db
from services.generate import generate_answer
import pickle
import os

from initialize import init_rag  # 우리가 만든 함수

app = FastAPI()

def initialize(
    data_dir="data/yongin_data2",
    use_existing_index=True, # 새로 만들기 여부
    chunk_strategy="token",  # 청킹 방식
    chunk_param=500,         # 청킹 사이즈
    index_type="HNSW",       # 인덱스 방식
    ranking_mode="rrf", 
    index_file="faiss_index.bin",
    chunked_file="chunked_data.pkl",
):
    print("Starting up...")

    if not use_existing_index:
        # 필요하다면 여기서 새로 인덱스 생성
        print("Re-building index from scratch...")
        init_rag(
            data_dir=data_dir,
            chunk_strategy=chunk_strategy,
            chunk_param=chunk_param,
            index_type=index_type,
            output_index_path=index_file,
            output_chunk_path=chunked_file
        )
    else:
        # 기존 인덱스와 chunk 정보 로딩
        db.load_data(data_dir)
        db.load_index(index_file, index_type=index_type)

        try:
            with open(chunked_file, "rb") as f:
                db.chunked_data = pickle.load(f)
                print(f"chunked_data loaded from {chunked_file}, total chunks: {len(db.chunked_data['all_chunks'])}")
        except FileNotFoundError:
            print(f"[Warning] {chunked_file} not found. Possibly you need to run initialize.py or set use_existing_index=False.")
        except Exception as e:
            print(f"[Error] Failed to load {chunked_file}: {e}")

        if db.index is None or db.index.ntotal == 0:
            raise ValueError("FAISS index failed to initialize!")
        if not db.documents:
            raise ValueError("No documents loaded!")
        if not db.chunked_data:
            print("[Warning] chunked_data is empty. Possibly you need to run initialize.py first?")

    print("Startup complete.")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    answer = generate_answer(query.question, top_k=5, ranking_mode=ranking_mode, llm_backend=llm_backend)
    return {"question": query.question, "answer": answer}

if __name__ == "__main__":
    """
    아래 파라미터들만 적절히 바꿔가면서 테스트 가능.
    CLI 인자나 config 파일로 바꿔도 됨.
    """
    data_dir = "data/yongin_data2"
    use_existing_index = True  # False로 바꾸면 매번 fresh 인덱스 빌드
    chunk_strategy = "recursive"  # "fixed", "token", "recursive"
    chunk_param = 800
    index_type = "FLAT"  # "FLAT" or "HNSW"
    ranking_mode = "rrf"  # "dense", "tfidf", "rrf" 중 하나
    llm_backend = "ollama_deepseek" # "openai" or "ollama_deepseek"
    initialize(
        data_dir=data_dir,
        use_existing_index=use_existing_index,
        chunk_strategy=chunk_strategy,
        chunk_param=chunk_param,
        index_type=index_type,
        ranking_mode=ranking_mode,
        index_file="faiss_index.bin",
        chunked_file="chunked_data.pkl"
    )

    uvicorn.run(app, host="127.0.0.1", port=8000)
