from typing import List, Tuple

import streamlit as st


def filter_results(results: List[dict]) -> List[Tuple[dict, str]]:
    """검색 결과 필터링"""
    filtered_results = []
    minibook_results = []
    other_results = []

    for r in results:
        if r.get("score", 0) >= 0.5 and len(r.get("content", "").strip()) > 30:
            book_name = r.get("book_name") or r.get("metadata", {}).get("title", "")
            if "미니북" in book_name:
                minibook_results.append((r, book_name))
            else:
                other_results.append((r, book_name))

    # 미니북이 없는 경우 점수가 낮거나 내용이 짧은 미니북도 포함
    if not minibook_results:
        for r in results:
            book_name = r.get("book_name") or r.get("metadata", {}).get("title", "")
            if "미니북" in book_name and len(r.get("content", "").strip()) > 0:
                minibook_results.append((r, book_name))
                break

    # 결과 조합: 미니북을 우선 포함하고 나머지 결과 추가
    filtered_results.extend(minibook_results)
    remaining_slots = 3 - len(filtered_results)  # 최대 3개까지만 유지
    if remaining_slots > 0:
        filtered_results.extend(other_results[:remaining_slots])

    return filtered_results


def process_search_results(
    results: List[dict], question_id: str
) -> Tuple[str, List[str]]:
    """검색 결과 처리 및 컨텍스트 생성"""
    context_chunks = []
    sources = []

    if not results:
        return "관련 내용을 찾지 못했습니다.", []

    filtered_results = filter_results(results)
    for result in filtered_results:
        r, book_name = result
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
