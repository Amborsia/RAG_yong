from typing import Any, Dict

import streamlit as st
from langchain_core.messages.chat import ChatMessage

from utils.greeting_message import GREETING_MESSAGE


def init_session_state():
    """세션 상태 초기화"""
    initial_states: Dict[str, Any] = {
        "messages": [
            ChatMessage(role="assistant", content=GREETING_MESSAGE["ebs_tutor"])
        ],
        "pdf_viewer_state": {"current_page": None},
        "search_results": None,
        "questions": {},  # 질문 목록
        "sources": {},  # 참고 페이지 정보
        "modal_open": False,  # 모달 상태
        "pdf_viewer": None,  # PDF 뷰어 인스턴스
        "modal_current_page": {},  # 모달 페이지 상태
        "active_question_id": None,
        "pdf_viewer_directories": {},  # 책 이름별 디렉토리
        "book_names": {},  # 질문 ID별 책 이름
        "question_results": {},  # 질문별 검색 결과
    }

    for key, value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
