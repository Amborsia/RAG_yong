# models/database.py
import os
import json
import faiss
import numpy as np

# 전역 변수
index = None            # FAISS 인덱스 (HNSW 등)
documents = []          # 원본 문서 (JSON 전체)
chunked_data = {}       # chunk 정보 저장 (all_chunks, chunk_to_doc_map 등)

def load_data(data_dir):
    """
    data_dir 내 JSON 파일들을 읽어 documents 리스트에 저장.
    각 JSON은 "url", "content" 필드를 포함해야 함.
    """
    global documents
    documents.clear()

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if "url" in data and "content" in data:
                        documents.append(data)
                    else:
                        print(f"[Warning] Missing fields in {file_name}.")
            except Exception as e:
                print(f"[Error] Failed to load {file_name}: {e}")

    print(f"Loaded {len(documents)} documents from '{data_dir}'.")

def build_index(embeddings, index_type="HNSW"):
    """
    index_type에 따라 다른 FAISS 인덱스를 생성.
    기본값 index_type="HNSW"로 HNSW 인덱스를 사용.
    """
    global index
    dimension = embeddings.shape[1]

    if index_type == "FLAT":
        # 기존 L2 Flat 인덱스
        idx = faiss.IndexFlatL2(dimension)
        idx.add(embeddings)
        index = idx
        return index

    elif index_type == "HNSW":
        # HNSW 인덱스 예시
        M = 32  # HNSW 그래프의 branching factor
        idx = faiss.IndexHNSWFlat(dimension, M)

        # 인덱스 생성시 검색 정확도(efConstruction)와 탐색 폭(efSearch) 조정
        idx.hnsw.efConstruction = 80
        idx.hnsw.efSearch = 64

        # 임베딩 추가
        idx.add(embeddings)
        index = idx
        print(f"HNSW index built with {embeddings.shape[0]} embeddings (M={M}).")
        return index

    else:
        print(f"[Error] Unknown index type: {index_type}")
        index = None
        return None

def load_index(file_path, index_type="HNSW"):
    """
    faiss.read_index로 인덱스를 로드.
    (HNSW / FLAT 등 어떤 인덱스인지에 따라 별도 설정이 필요한 경우 반영)
    """
    global index
    try:
        idx = faiss.read_index(file_path)
        if isinstance(idx, faiss.IndexHNSWFlat) and index_type == "HNSW":
            # efSearch 파라미터를 다시 설정할 수 있음
            idx.hnsw.efSearch = 64
            index = idx
        else:
            # FLAT 인덱스거나 다른 타입이면 그대로 할당
            index = idx

        print(f"FAISS index loaded from {file_path}, ntotal = {index.ntotal}")
        return index

    except FileNotFoundError:
        print(f"[Error] Index file not found: {file_path}")
    except Exception as e:
        print(f"[Error] Failed to load index from {file_path}: {e}")
    index = None
    return None

def search_embeddings(idx, query_embedding, top_k=5):
    """
    FAISS 인덱스에서 쿼리 임베딩과 가까운 top_k 개 chunk의 (distance, index) 반환
    """
    if idx is None:
        raise ValueError("FAISS index is not initialized.")
    distances, indices = idx.search(query_embedding.reshape(1, -1), k=top_k)
    return distances, indices
