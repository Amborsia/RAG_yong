import uuid

import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI

from services.ebs import EbsRAG
from utils.prompts import load_prompt

from .search import process_search_results


def handle_user_input(user_input: str, ebs_rag: EbsRAG):
    """사용자 입력 처리"""
    question_id = str(uuid.uuid4())
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    st.session_state["questions"][question_id] = user_input

    try:
        results = ebs_rag.search(user_input, top_k=3)
        st.session_state["search_results"] = results
        st.session_state["question_results"][question_id] = results

        context_text, sources = process_search_results(results, question_id)

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

        if "관련 자료를 찾을 수 없습니다" in response_text:
            sources = [{"message": "생성된 답변입니다"}]

        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response_text)
        )
        st.session_state["sources"][question_id] = sources
        if results:
            st.session_state["pdf_viewer_state"]["current_page"] = results[0]["page_no"]

    except Exception as e:
        st.error(f"검색 중 오류 발생: {str(e)}")
