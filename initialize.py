# initialize.py

import numpy as np
import faiss
import os
import models.database as db
from models.embedding import encode_texts
from utils.chunking import (
    token_based_chunking,
    fixed_size_chunking,
    recursive_chunking,
)
import pickle

# initialize.py

def init_rag(
    data_dir="crawling/output",  # app.py에서 DATA_DIR로 설정한 경로와 일치시킴
    chunk_strategy="token",
    chunk_param=500,
    index_type="HNSW",
    output_index_path="rag_index/index.faiss",  # app.py와 일치
    output_chunk_path="rag_index/index.pkl"     # app.py와 일치
):
    print(f"🔍 init_rag() 호출됨! (chunk_strategy={chunk_strategy}, chunk_param={chunk_param}, index_type={index_type})")
    # 1) 문서 로드
    db.load_data(data_dir)
    if not db.documents:
        raise ValueError("No documents loaded. Check the data directory.")
    print(f"Documents loaded: {len(db.documents)}")

    # 2) chunk 분할
    all_chunks = []
    chunk_to_doc_map = []

    if chunk_strategy == "fixed":
        chunk_fn = lambda text: fixed_size_chunking(text, chunk_size=chunk_param)
    elif chunk_strategy == "recursive":
        chunk_fn = lambda text: recursive_chunking(text, max_tokens=chunk_param)
    else:  # 기본값 "token"
        chunk_fn = lambda text: token_based_chunking(text, max_tokens=chunk_param)

    for doc_idx, doc in enumerate(db.documents):
        # 새 구조에서는 문서의 텍스트는 doc["text"]
        content = doc.get("text", "")
        # 메타데이터로 URL 등이 포함될 수 있지만, 여기서는 단순히 텍스트만 사용
        chunks = chunk_fn(content)
        for ch in chunks:
            all_chunks.append(ch)
            chunk_to_doc_map.append(doc_idx)

    print(f"Total chunks created: {len(all_chunks)}")

    # 3) chunk 임베딩
    chunk_embeddings = encode_texts(all_chunks, batch_size=10)
    print(f"Generated {len(chunk_embeddings)} chunk embeddings.")
    if len(all_chunks) != len(chunk_embeddings):
        print(f"[Warning] Mismatch: {len(all_chunks)} chunks, {len(chunk_embeddings)} embeddings.")

    # 4) 인덱스 생성
    index = db.build_index(chunk_embeddings, index_type=index_type)
    if index is None or index.ntotal == 0:
        raise ValueError("FAISS index creation failed or is empty.")
    print(f"FAISS index built with {index.ntotal} chunk embeddings.")

    # 5) 인덱스, chunk 데이터 파일로 저장
    # 폴더 "rag_index"가 없으면 생성
    if not os.path.exists("rag_index"):
        os.makedirs("rag_index")
    faiss.write_index(index, output_index_path)
    print(f"FAISS index saved to {output_index_path}.")

    db.chunked_data = {
        "all_chunks": all_chunks,
        "chunk_to_doc_map": chunk_to_doc_map,
    }

    with open(output_chunk_path, "wb") as f:
        pickle.dump(db.chunked_data, f)
    print(f"chunked_data saved to {output_chunk_path}")



def main():
    """
    기존처럼 단독 실행할 때,
    인자를 직접 바꿔보고 싶다면 아래 부분 수정
    예) CLI 인자 파싱, sys.argv, argparse 등으로 확장 가능
    """
    init_rag(
        data_dir="data/yongin_data2",
        chunk_strategy="token",  # "fixed", "recursive", "token"
        chunk_param=500,
        index_type="HNSW",       # "FLAT" or "HNSW"
        output_index_path="faiss_index.bin",
        output_chunk_path="chunked_data.pkl"
    )

if __name__ == "__main__":
    main()
