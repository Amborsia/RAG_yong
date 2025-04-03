import uuid

import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI

from services.constants import NOT_FOUND_IN_TEXTBOOK
from services.ebs import EbsRAG
from utils.prompts import load_prompt


def handle_user_input(user_input: str, ebs_rag: EbsRAG):
    """사용자 입력 처리"""
    question_id = str(uuid.uuid4())
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    st.session_state["questions"][question_id] = user_input

    try:
        with st.spinner("검색 중..."):
            # 검색과 결과 가공을 한번에 처리
            context_text, sources, results = ebs_rag.search_with_processed_results(
                user_input, question_id, top_k=3
            )

            # 검색 결과 저장
            st.session_state["search_results"] = results
            st.session_state["question_results"][question_id] = results

        # LLM 응답 생성
        prompt_template = load_prompt("prompts/ebs_tutor.yaml")

        # 채팅 기록 구성 (최근 3회차 대화 포함)
        chat_history = "\n".join(
            [
                f"{msg.role}: {msg.content}"
                for msg in st.session_state["messages"][-6:-1]
            ]  # 최근 3회차(사용자+어시스턴트 메시지 3세트)
        )

        formatted_prompt = prompt_template.format(
            context_text=context_text, question=user_input, chat_history=chat_history
        )

        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
        response_generator = llm.stream(formatted_prompt)

        response_text = ""
        with st.chat_message("assistant"):
            response_container = st.empty()
            for chunk in response_generator:
                chunk_text = getattr(chunk, "content", str(chunk))
                response_text += chunk_text
                response_container.markdown(response_text)

        if "찾을 수 없는 내용이에요" in response_text:
            sources = [{"message": "생성된 답변입니다"}]

        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response_text)
        )
        st.session_state["sources"][question_id] = sources
        if sources:  # 소스가 있는 경우에만 페이지 업데이트
            try:
                first_source = sources[0]
                if isinstance(first_source, str):
                    page_no = int(first_source.replace("페이지", ""))
                    st.session_state["pdf_viewer_state"]["current_page"] = page_no
            except (ValueError, IndexError):
                pass  # 페이지 번호 파싱 실패 시 무시

    except Exception as e:
        st.error(f"검색 중 오류 발생: {str(e)}")
