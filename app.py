# app.py
import json
import logging
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI

from services.ebs import EbsRAG

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 스타일 설정
st.markdown(
    """
<style>
.stMain {
    position: relative;
}
.stChatMessage {
    background-color: transparent !important;
}

/* 채팅 입력창 높이 조정 */
.st-emotion-cache-glsyku {
    min-height: 80px !important;
    align-items: center;
}

/* 사용자 메시지 스타일링 */
[data-testid="stChatMessage"]:has(> [data-testid="stChatMessageAvatarUser"]) {
    justify-content: flex-end !important;
    display: flex !important;
}
[data-testid="stChatMessage"]:has(> [data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    text-align: right !important;
    background-color: #3399FF !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    padding: 10px !important;
    margin: 5px 0 !important;
    max-width: 80% !important;
    flex-grow: 0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# 타이틀
st.title("EBS 과학 튜터 챗봇")

# 초기 인사말
GREETING_MESSAGE = """안녕하세요! 저는 EBS 중학 과학 1 교재를 기반으로 학습을 도와주는 튜터입니다. 
교재 내용 중에서 궁금한 점이 있다면 무엇이든 물어보세요."""


def load_prompt(file_path: str, encoding: str = "utf-8") -> str:
    with open(file_path, encoding=encoding) as f:
        return f.read()


# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content=GREETING_MESSAGE)
    ]

# PDF 뷰어 관련 상태 관리
if "pdf_viewer_state" not in st.session_state:
    st.session_state["pdf_viewer_state"] = {
        "current_page": None,
        "highlight_text": None,
    }

# 레이아웃 구성
chat_col, pdf_col = st.columns([3, 2])

with chat_col:
    # 기존 채팅 인터페이스
    for message in st.session_state["messages"]:
        st.chat_message(message.role).write(message.content)

with pdf_col:
    # PDF 뷰어 플레이스홀더 (향후 PDF 뷰어 라이브러리 연동 예정)
    if st.session_state["pdf_viewer_state"]["current_page"]:
        st.info("PDF 페이지 표시 영역")
        metadata = st.session_state["pdf_viewer_state"]["current_page"]
        st.write(f"현재 페이지: {metadata}")

# 채팅 응답 생성 부분
if user_input := st.chat_input("궁금한 내용을 물어보세요!"):
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

    # 검색 수행
    ebs_rag = EbsRAG()
    results = ebs_rag.search(user_input, top_k=3)

    # 컨텍스트 구성 (출처 정보 포함)
    if results:
        context_chunks = []
        sources = []  # 출처 정보 저장
        for r in results:
            page_no = r["page_no"]
            content = r["content"]
            context_chunks.append(f"[{page_no}페이지]\n{content}")
            sources.append(f"- {page_no}페이지")

        context_text = "\n\n".join(context_chunks)

        # 현재 페이지 업데이트 (PDF 뷰어용)
        st.session_state["pdf_viewer_state"]["current_page"] = results[0]["page_no"]
    else:
        context_text = "관련 내용을 찾지 못했습니다."
        sources = []

    # GPT 응답 생성
    prompt_template = load_prompt("prompts/ebs_tutor.yaml")
    formatted_prompt = prompt_template.format(
        context_text=context_text, question=user_input
    )

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
    response_generator = llm.stream(formatted_prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        response_text = ""

        # 응답 생성 중 표시
        spinner_placeholder = st.empty()
        spinner_placeholder.markdown("**답변 생성 중입니다...**")

        for chunk in response_generator:
            if response_text == "":
                spinner_placeholder.empty()

            if hasattr(chunk, "content"):
                chunk_text = chunk.content
            else:
                chunk_text = str(chunk)

            response_text += chunk_text
            response_container.markdown(response_text)

        # 출처 정보 표시
        if sources:
            st.markdown("**참고 페이지:**")
            for source in sources:
                st.markdown(source)

    # 응답 저장
    st.session_state["messages"].append(
        ChatMessage(role="assistant", content=response_text)
    )

# PDF 뷰어 영역 표시 (사이드바 또는 별도 컬럼)
if st.session_state["pdf_viewer_state"]["current_page"]:
    st.sidebar.markdown(
        f"**현재 페이지:** {st.session_state['pdf_viewer_state']['current_page']}"
    )

    # 검색 결과 디버깅
    if results:
        with st.expander("디버깅 정보"):
            st.write("검색 결과:", results)
            st.write(
                "현재 PDF 페이지:", st.session_state["pdf_viewer_state"]["current_page"]
            )

    # 주요 지점에 로깅 추가
    logger.info(f"검색 쿼리: {user_input}")
    logger.info(f"검색 결과 수: {len(results)}")
