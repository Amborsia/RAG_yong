# initialize.py

import numpy as np
import faiss
import models.database as db
from models.embedding import encode_texts
from utils.chunking import (
    token_based_chunking,
    fixed_size_chunking,
    recursive_chunking,
)
import pickle

def init_rag(
    data_dir="data/yongin_data2",
    chunk_strategy="token",
    chunk_param=500,
    index_type="HNSW",
    output_index_path="faiss_index.bin",
    output_chunk_path="chunked_data.pkl"
):
    """
    RAG 시스템 초기화 과정을 수행:
      1) 문서 로드
      2) 문서 → chunk 분할 (chunk_strategy에 따라)
      3) chunk 임베딩
      4) FAISS 인덱스 생성 (index_type에 따라)
      5) 인덱스, chunk 정보 저장
    """

    # 1) 문서 로드
    db.load_data(data_dir)
    if not db.documents:
        raise ValueError("No documents loaded. Check the data directory.")
    print(f"Documents loaded: {len(db.documents)}")

    # 2) chunk 분할
    all_chunks = []
    chunk_to_doc_map = []

    # chunking 함수를 전략별로 매핑
    if chunk_strategy == "fixed":
        chunk_fn = lambda text: fixed_size_chunking(text, chunk_size=chunk_param)
    elif chunk_strategy == "recursive":
        chunk_fn = lambda text: recursive_chunking(text, max_tokens=chunk_param)
    else:  # 기본값 "token"
        chunk_fn = lambda text: token_based_chunking(text, max_tokens=chunk_param)

    for doc_idx, doc in enumerate(db.documents):
        content = doc.get("content", "")
        url = doc.get("url", "")
        combined_text = f"URL: {url}\n{content}"
        chunks = chunk_fn(combined_text)
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
    faiss.write_index(index, output_index_path)
    print(f"FAISS index saved to {output_index_path}.")

    db.chunked_data = {
        "all_chunks": all_chunks,
        "chunk_to_doc_map": chunk_to_doc_map,
    }

    with open(output_index_path, "wb") as fidx:
        faiss.write_index(index, output_index_path)

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
