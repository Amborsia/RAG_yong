from typing import List, Tuple

import streamlit as st


def filter_results(results: List[dict]) -> List[Tuple[dict, str]]:
    """검색 결과 필터링"""
    filtered_results = []
    for r in results:
        if r.get("score", 0) >= 0.5 and len(r.get("content", "").strip()) > 30:
            book_name = r.get("book_name") or r.get("metadata", {}).get("title")
            filtered_results.append((r, book_name))
    return filtered_results


def process_search_results(
    results: List[dict], question_id: str
) -> Tuple[str, List[str]]:
    """검색 결과 처리 및 컨텍스트 생성"""
    context_chunks = []
    sources = []

    if not results:
        return "관련 내용을 찾지 못했습니다.", []

    for r, book_name in filter_results(results):
        page_no = r.get("page_no")
        content = r.get("content")
        if page_no and content:
            context_chunks.append(f"[{page_no}페이지]\n{content}")
            sources.append(f"{page_no}페이지")
        st.session_state["book_names"][question_id] = book_name

    return (
        "\n\n".join(context_chunks)
        if context_chunks
        else "관련 내용을 찾지 못했습니다."
    ), sources
