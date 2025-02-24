# services/search.py
import numpy as np
import models.database as db
from models.embedding import encode_texts
from utils.chunking import sort_chunks_by_similarity

def get_chunked_data():
    """
    db.chunked_data가 dict 형태가 아니라면, 변환을 시도합니다.
    만약 tuple 또는 InMemoryDocstore 같은 객체라면, 첫번째 요소 또는 내부 docs를 dict로 변환합니다.
    """
    data = db.chunked_data
    # 이미 dict인 경우 그대로 반환
    if isinstance(data, dict):
        return data
    # 만약 tuple이면 첫 번째 요소를 사용 (예: (dict, ...))
    if isinstance(data, tuple) and len(data) > 0 and isinstance(data[0], dict):
        db.chunked_data = data[0]
        return db.chunked_data
    # InMemoryDocstore 등 내부 docs 속성이 있는 경우
    if hasattr(data, "docs") and isinstance(data.docs, dict):
        db.chunked_data = data.docs
        return db.chunked_data
    # 변환 실패 시 빈 dict 반환
    return {}

def reciprocal_rank_fusion(results_list, k=60):
    """
    Reciprocal Rank Fusion (RRF)로 여러 랭킹 목록을 합치는 함수.
    results_list: [
       [(chunk_idx, rank), (chunk_idx, rank), ...],  # dense_results
       [(chunk_idx, rank), (chunk_idx, rank), ...],  # tfidf_results
       ... (필요 시 더 추가 가능)
    ]
    k: RRF 공식에서 사용하는 상수 (기본 60)
    """
    all_chunks = set()
    for results in results_list:
        for (c_idx, _) in results:
            all_chunks.add(c_idx)

    scores = {}
    for c in all_chunks:
        score_sum = 0.0
        for res in results_list:
            found_rank = None
            for (c_idx, rnk) in res:
                if c_idx == c:
                    found_rank = rnk
                    break
            if found_rank is not None:
                score_sum += 1.0 / (k + found_rank)
        scores[c] = score_sum

    fused_ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused_ranking

def get_dense_ranking(query, top_k=5):
    """
    FAISS Dense 검색에서 나온 결과를 (chunk_idx, rank) 형태로 반환.
    rank는 1부터 시작.
    """
    if db.index is None:
        raise ValueError("FAISS index not loaded or initialized.")

    query_vec = encode_texts([query]).astype(np.float32)
    distances, indices = db.search_embeddings(db.index, query_vec[0], top_k=top_k)

    results = []
    for rank_in_list, chunk_idx in enumerate(indices[0]):
        results.append((int(chunk_idx), rank_in_list + 1))
    return results

def get_tfidf_ranking(query, top_k=5):
    """
    TF-IDF 기반으로 모든 chunk에 대해 쿼리와 유사도를 계산하여,
    상위 top_k 개 (chunk_idx, rank)를 반환.
    """
    chunked = get_chunked_data()
    all_chunks = chunked.get("all_chunks", [])
    if not all_chunks:
        return []

    # 기존 로직: chunk_texts 복사 후 정렬.
    chunk_texts = all_chunks[:]
    sorted_by_tfidf = sort_chunks_by_similarity(chunk_texts, query)
    ranked_list = []
    for rank_in_list, chunk_text in enumerate(sorted_by_tfidf):
        # 동일 텍스트가 여러 개 있는 경우 index()는 첫 번째 인덱스만 반환함에 주의.
        chunk_idx = chunk_texts.index(chunk_text)
        ranked_list.append((chunk_idx, rank_in_list + 1))
    return ranked_list[:top_k]

def search_top_k(query, top_k=5, ranking_mode="rrf"):
    """
    ranking_mode에 따라 검색 접근:
       - 'rrf' : Dense + TF-IDF RRF 결합.
       - 'dense': Dense(Faiss) 결과만.
       - 'tfidf': TF-IDF 결과만.
    최종 상위 top_k chunk 반환.
    """
    if ranking_mode == "rrf":
        dense_results = get_dense_ranking(query, top_k=top_k)
        tfidf_results = get_tfidf_ranking(query, top_k=top_k)
        rrf_merged = reciprocal_rank_fusion([dense_results, tfidf_results], k=60)
        top_rrf = rrf_merged[:top_k]
        final_chunk_indices = [x[0] for x in top_rrf]

    elif ranking_mode == "dense":
        dense_results = get_dense_ranking(query, top_k=top_k)
        final_chunk_indices = [x[0] for x in dense_results]

    elif ranking_mode == "tfidf":
        tfidf_results = get_tfidf_ranking(query, top_k=top_k)
        final_chunk_indices = [x[0] for x in tfidf_results]

    else:
        print(f"[Warning] Unknown ranking_mode={ranking_mode}, defaulting to 'rrf'")
        dense_results = get_dense_ranking(query, top_k=top_k)
        tfidf_results = get_tfidf_ranking(query, top_k=top_k)
        rrf_merged = reciprocal_rank_fusion([dense_results, tfidf_results], k=60)
        top_rrf = rrf_merged[:top_k]
        final_chunk_indices = [x[0] for x in top_rrf]

    # 최종적으로 chunked_data에서 데이터를 읽어옴.
    chunked = get_chunked_data()
    print("Chunked data", chunked)
    all_chunks = chunked.get("all_chunks", [])
    print("All Chunks", all_chunks)
    chunk_to_doc_map = chunked.get("chunk_to_doc_map", [])
    print("Chunk to Doc Map", chunk_to_doc_map)

    results = []
    for c_idx in final_chunk_indices:
        if c_idx >= len(all_chunks):
            continue
        chunk_text = all_chunks[c_idx]
        doc_idx = chunk_to_doc_map[c_idx]
        doc_data = db.documents[doc_idx]
        results.append({
            "chunk_text": chunk_text,
            "doc_idx": doc_idx,
            "original_doc": doc_data
        })

    return results