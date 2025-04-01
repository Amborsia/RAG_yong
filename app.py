import logging

import streamlit as st
from dotenv import load_dotenv

from services.ebs import EbsRAG
from services.handlers.chat import handle_user_input
from services.state.session import init_session_state
from services.ui.pdf_modal import pdf_viewer_modal
from services.ui.sidebar import render_sidebar
from services.ui.styles import apply_custom_styles

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()


def main():
    """메인 애플리케이션"""
    # 초기화
    init_session_state()
    apply_custom_styles()

    # EBS 검색 객체 생성
    ebs_rag = EbsRAG()

    # 메인 UI
    st.title("EBS 과학 튜터 챗봇")

    # 기존 대화 표시
    for message in st.session_state["messages"]:
        st.chat_message(message.role).write(message.content)

    # 사용자 입력 처리
    if user_input := st.chat_input("궁금한 내용을 물어보세요!", key="chat_input"):
        handle_user_input(user_input, ebs_rag)

    # 사이드바 렌더링
    render_sidebar()

    # 모달 처리
    if st.session_state.get("active_question_id") is not None:
        pdf_viewer_modal(st.session_state["active_question_id"])
        st.session_state["active_question_id"] = None


if __name__ == "__main__":
    main()
