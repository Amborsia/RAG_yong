import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI

from services.constants import CONTENT_NOT_IN_TEXTBOOK
from services.ebs import EbsRAG
from utils.llm.gemma import GemmaLLM, create_user_message
from utils.prompts import load_prompt

# 환경 변수 로드
load_dotenv()

# Gemma3 설정
GEMMA_URL = os.getenv("GEMMA_URL", "http://localhost:8000")  # 기본값 설정
llm_type = os.getenv("LLM_TYPE", "openai")


def get_llm() -> Union[GemmaLLM, ChatOpenAI]:
    """환경 설정에 따른 LLM 인스턴스 반환"""
    if llm_type == "gemma":
        return GemmaLLM.from_env(project_name="ebs-tutor")
    elif llm_type == "openai":
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    else:
        raise ValueError(f"Invalid LLM type: {llm_type}")


def process_gemma_stream(
    response: Any,
    response_container: Any,
    context_container: Optional[Any] = None,
    show_response: bool = True,
) -> str:
    """Gemma 스트리밍 응답 처리"""
    response_text = ""
    for line in response.iter_lines():
        if not line:
            continue

        try:
            line_text = line.decode("utf-8")
            if not line_text.startswith("data: "):
                continue

            line_text = line_text[6:]
            if line_text.strip() == "[DONE]":
                break

            try:
                json_response = json.loads(line_text)
                if not (
                    json_response.get("choices") and len(json_response["choices"]) > 0
                ):
                    continue

                delta = json_response["choices"][0].get("delta", {})
                if delta.get("content"):
                    response_text += delta["content"]
                    if context_container and response_text.strip():
                        context_container.empty()
                    if show_response:
                        response_container.markdown(response_text + "▌")
            except json.JSONDecodeError:
                continue
        except UnicodeDecodeError:
            continue

    return response_text


def process_openai_stream(
    response: Any,
    response_container: Any,
    context_container: Optional[Any] = None,
    show_response: bool = True,
) -> str:
    """OpenAI 스트리밍 응답 처리"""
    response_text = ""
    for chunk in response:
        chunk_text = getattr(chunk, "content", str(chunk))
        response_text += chunk_text
        if context_container and response_text.strip():
            context_container.empty()
        if show_response:
            response_container.markdown(response_text + "▌")
    return response_text


def process_context_check(
    llm: Union[GemmaLLM, ChatOpenAI], prompt: str
) -> Tuple[bool, str]:
    """컨텍스트 체크 수행 및 결과 반환"""
    check_messages = [create_user_message(prompt)] if llm_type == "gemma" else prompt
    check_response = llm.stream(check_messages)

    # 스트리밍 응답 처리 (화면에 표시하지 않음)
    response_text = ""
    if isinstance(llm, GemmaLLM):
        response_text = process_gemma_stream(
            check_response, st.empty(), show_response=False
        )
    else:
        response_text = process_openai_stream(
            check_response, st.empty(), show_response=False
        )

    # 응답 파싱
    check_lines = response_text.strip().split("\n", 1)
    can_answer = check_lines[0].strip() == "Yes"

    return can_answer, response_text


def generate_final_response(
    llm: Union[GemmaLLM, ChatOpenAI],
    prompt: str,
    can_answer: bool,
    response_container: Any,
    context_container: Any,
) -> Tuple[str, List[Dict[str, str]]]:
    """최종 응답 생성"""
    messages = [create_user_message(prompt)] if llm_type == "gemma" else prompt
    response = llm.stream(messages)

    # 스트리밍 응답 처리
    if isinstance(llm, GemmaLLM):
        response_text = process_gemma_stream(
            response, response_container, context_container
        )
    else:
        response_text = process_openai_stream(
            response, response_container, context_container
        )

    # 최종 응답 표시
    if response_text:
        response_container.markdown(response_text)
        sources = [{"message": CONTENT_NOT_IN_TEXTBOOK}] if not can_answer else []
    else:
        response_text = "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
        sources = []

    return response_text, sources


def handle_user_input(user_input: str, ebs_rag: EbsRAG):
    """사용자 입력 처리"""
    question_id = str(uuid.uuid4())
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    st.session_state["questions"][question_id] = user_input

    try:
        # 1-1단계: 컨텍스트 검색
        with st.spinner("관련 교재 내용을 검색 중입니다..."):
            context_text, sources, results = ebs_rag.search_with_processed_results(
                user_input, question_id, top_k=3
            )
            st.session_state["search_results"] = results
            st.session_state["question_results"][question_id] = results

        # 1-2단계: 컨텍스트 분석
        with st.spinner("검색된 내용의 관련성을 분석 중입니다..."):
            # 컨텍스트 체크 프롬프트 준비
            context_check_prompt = load_prompt(
                "prompts/ebs_tutor.yaml", prompt_name="context_check"
            )
            formatted_check_prompt = context_check_prompt.format(
                context_text=context_text,
                question=user_input,
            )

            # LLM으로 컨텍스트 체크
            llm = get_llm()
            can_answer, _ = process_context_check(llm, formatted_check_prompt)

            # can_answer 값을 세션 상태에 저장
            if "can_answer" not in st.session_state:
                st.session_state["can_answer"] = {}
            st.session_state["can_answer"][question_id] = can_answer

        # 2단계: 최종 응답 생성
        with st.chat_message("assistant"):
            response_container = st.empty()
            context_container = st.empty()

            try:
                # 튜터 응답 프롬프트 준비
                prompt_name = "context_based_answer" if can_answer else "general_answer"
                tutor_prompt = load_prompt(
                    "prompts/ebs_tutor.yaml", prompt_name=prompt_name
                )

                # 채팅 기록 구성
                chat_history = "\n".join(
                    [
                        f"{msg.role}: {msg.content}"
                        for msg in st.session_state["messages"][-6:-1]
                    ]
                )

                # 튜터 응답 프롬프트 포맷
                formatted_tutor_prompt = tutor_prompt.format(
                    context_text=context_text if can_answer else "",
                    question=user_input,
                    chat_history=chat_history if can_answer else "",
                )

                # 최종 응답 생성
                response_text, sources = generate_final_response(
                    llm,
                    formatted_tutor_prompt,
                    can_answer,
                    response_container,
                    context_container,
                )

            except Exception as e:
                st.error(f"LLM 응답 생성 중 오류 발생: {str(e)}")
                response_text = "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
                sources = []

        # 응답 및 소스 저장
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response_text)
        )
        st.session_state["sources"][question_id] = sources

        # PDF 뷰어 페이지 업데이트
        if can_answer and sources:
            try:
                first_source = sources[0]
                if isinstance(first_source, str):
                    page_no = int(first_source.replace("페이지", ""))
                    st.session_state["pdf_viewer_state"]["current_page"] = page_no
            except (ValueError, IndexError):
                pass

    except Exception as e:
        st.error(f"검색 중 오류 발생: {str(e)}")
