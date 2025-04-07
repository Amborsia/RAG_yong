# models/database.py
import json
import os

import faiss
import numpy as np

# 전역 변수
index = None  # FAISS 인덱스 (HNSW 등)
documents = []  # 원본 문서 리스트 (각 JSON 파일의 내용)
chunked_data = {}  # chunk 정보 저장 (all_chunks, chunk_to_doc_map 등)


def load_data(data_dir):
    """
    data_dir 내 JSON 파일들을 읽어 documents 리스트에 저장.
    각 JSON 파일은 {"title": str, "pages": Dict[str, str]} 형식이어야 합니다.
    """
    global documents
    documents.clear()

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    if isinstance(data, dict) and "title" in data and "pages" in data:
                        # 각 페이지를 개별 문서로 변환
                        for page_no, content in data["pages"].items():
                            if content:  # 빈 페이지는 제외
                                doc = {
                                    "text": content,
                                    "metadata": {
                                        "title": data["title"],
                                        "page": page_no,
                                    },
                                }
                                documents.append(doc)
                    else:
                        print(
                            f"[Warning] Invalid JSON structure in {file_name}. Expected 'title' and 'pages' fields."
                        )
            except Exception as e:
                print(f"[Error] Failed to load {file_name}: {e}")

    print(f"Loaded {len(documents)} documents from '{data_dir}'.")


def build_index(embeddings, index_type="HNSW"):
    """
    주어진 임베딩(embeddings)으로 FAISS 인덱스를 생성.
    index_type에 따라 다른 인덱스를 생성하며, 기본값은 "HNSW"입니다.
    """
    global index
    dimension = embeddings.shape[1]

    if index_type == "FLAT":
        idx = faiss.IndexFlatL2(dimension)
        idx.add(embeddings)
        index = idx
        return index

    elif index_type == "HNSW":
        M = 32  # HNSW 그래프의 branching factor
        idx = faiss.IndexHNSWFlat(dimension, M)
        idx.hnsw.efConstruction = 80
        idx.hnsw.efSearch = 64
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
    faiss.read_index를 사용하여 인덱스를 로드.
    인덱스 타입에 따라 필요한 설정(예: HNSW의 efSearch)을 재설정합니다.
    """
    global index
    try:
        idx = faiss.read_index(file_path)
        if isinstance(idx, faiss.IndexHNSWFlat) and index_type == "HNSW":
            idx.hnsw.efSearch = 64
            index = idx
        else:
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
    FAISS 인덱스에서 쿼리 임베딩(query_embedding)과 가까운 top_k 개의 항목을 검색하여
    (distance, index) 튜플들을 반환합니다.
    """
    if idx is None:
        raise ValueError("FAISS index is not initialized.")
    distances, indices = idx.search(query_embedding.reshape(1, -1), k=top_k)
    return distances, indices
