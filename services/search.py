# services/search.py
import numpy as np
import models.database as db
from models.embedding import encode_texts
from utils.chunking import sort_chunks_by_similarity

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
    # 모든 chunk_idx를 수집
    all_chunks = set()
    for results in results_list:
        for (c_idx, _) in results:
            all_chunks.add(c_idx)

    # RRF 점수를 합산
    scores = {}
    for c in all_chunks:
        score_sum = 0.0
        for res in results_list:
            # 해당 랭킹 목록에서 c의 순위 찾기
            found_rank = None
            for (c_idx, rnk) in res:
                if c_idx == c:
                    found_rank = rnk
                    break
            if found_rank is not None:
                score_sum += 1.0 / (k + found_rank)
        scores[c] = score_sum

    # 점수 순으로 내림차순 정렬
    fused_ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused_ranking

def get_dense_ranking(query, top_k=5):
    """
    FAISS Dense 검색에서 나온 결과를 (chunk_idx, rank) 형태로 반환
    rank는 1부터 시작
    """
    if db.index is None:
        raise ValueError("FAISS index not loaded or initialized.")

    query_vec = encode_texts([query]).astype(np.float32)
    distances, indices = db.search_embeddings(db.index, query_vec[0], top_k=top_k)

    # indices[0] 내에 chunk 인덱스가 들어있음
    # 0번째가 가장 유사, 1번째가 그 다음 등등
    results = []
    for rank_in_list, chunk_idx in enumerate(indices[0]):
        results.append((int(chunk_idx), rank_in_list + 1))  # 1-based rank
    return results

def get_tfidf_ranking(query, top_k=5):
    """
    TF-IDF 기반으로 chunk 전체에 대해 쿼리와 유사도를 구해
    상위 top_k 개 (chunk_idx, rank)를 반환
    """
    all_chunks = db.chunked_data.get("all_chunks", [])
    if not all_chunks:
        return []

    # sort_chunks_by_similarity 함수는 chunk 텍스트만 정렬해서 반환하므로
    # chunk별 인덱스가 필요하다. 따라서 아래처럼 정렬 순서를 재구성.
    # 1) chunks를 복사
    chunk_texts = all_chunks[:]
    # 2) 정렬된 chunk_text 리스트를 얻는다
    sorted_by_tfidf = sort_chunks_by_similarity(chunk_texts, query)
    # 3) 해당 순서대로 chunk_idx를 매핑
    #    첫 번째가 TF-IDF 기준 rank=1, 두 번째가 rank=2 ...
    ranked_list = []
    for rank_in_list, chunk_text in enumerate(sorted_by_tfidf):
        chunk_idx = chunk_texts.index(chunk_text)
        ranked_list.append((chunk_idx, rank_in_list + 1))

    # 상위 top_k만 반환
    return ranked_list[:top_k]

def search_top_k(query, top_k=5, ranking_mode="rrf"):
    """
    1) ranking_mode에 따라 다양한 검색 접근:
       - 'rrf' : dense + tfidf RRF 결합
       - 'dense': dense(Faiss) 결과만
       - 'tfidf': tf-idf 결과만
    2) 최종 상위 top_k chunk 반환
    """
    if ranking_mode == "rrf":
        # 기존 로직: dense + tfidf 결합
        dense_results = get_dense_ranking(query, top_k=top_k)
        tfidf_results = get_tfidf_ranking(query, top_k=top_k)
        rrf_merged = reciprocal_rank_fusion([dense_results, tfidf_results], k=60)
        top_rrf = rrf_merged[:top_k]
        final_chunk_indices = [x[0] for x in top_rrf]

    elif ranking_mode == "dense":
        # Dense(Faiss)만 사용
        dense_results = get_dense_ranking(query, top_k=top_k)
        # dense_results는 [(chunk_idx, rank), ...] 형태
        # rank 순서대로 chunk_idx만 추출
        final_chunk_indices = [x[0] for x in dense_results]

    elif ranking_mode == "tfidf":
        # TF-IDF만 사용
        tfidf_results = get_tfidf_ranking(query, top_k=top_k)
        final_chunk_indices = [x[0] for x in tfidf_results]

    else:
        print(f"[Warning] Unknown ranking_mode={ranking_mode}, defaulting to 'rrf'")
        dense_results = get_dense_ranking(query, top_k=top_k)
        tfidf_results = get_tfidf_ranking(query, top_k=top_k)
        rrf_merged = reciprocal_rank_fusion([dense_results, tfidf_results], k=60)
        top_rrf = rrf_merged[:top_k]
        final_chunk_indices = [x[0] for x in top_rrf]

    # 이제 final_chunk_indices에 chunk 인덱스 목록이 있음
    all_chunks = db.chunked_data.get("all_chunks", [])
    chunk_to_doc_map = db.chunked_data.get("chunk_to_doc_map", [])

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

