import json
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage

from services.constants import CONTENT_NOT_IN_TEXTBOOK
from services.ebs import EbsRAG
from utils.llm.gemma import GemmaLLM, create_user_message
from utils.prompts import load_prompt

# 환경 변수 로드
load_dotenv()

# Gemma3 설정
GEMMA_URL = os.getenv("GEMMA_URL", "http://localhost:8000")  # 기본값 설정


def handle_user_input(user_input: str, ebs_rag: EbsRAG):
    """사용자 입력 처리"""
    question_id = str(uuid.uuid4())
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    st.session_state["questions"][question_id] = user_input

    try:
        # 1-1단계: 컨텍스트 검색
        with st.spinner("관련 교재 내용을 검색 중입니다..."):
            # 검색과 결과 가공을 한번에 처리
            context_text, sources, results = ebs_rag.search_with_processed_results(
                user_input, question_id, top_k=3
            )

            # 검색 결과 저장
            st.session_state["search_results"] = results
            st.session_state["question_results"][question_id] = results

        # 1-2단계: 컨텍스트 분석
        with st.spinner("검색된 내용의 관련성을 분석 중입니다..."):
            # 컨텍스트 체크 프롬프트 로드
            context_check_prompt = load_prompt(
                "prompts/ebs_tutor.yaml", prompt_name="context_check"
            )

            # 컨텍스트 체크 프롬프트 포맷
            formatted_check_prompt = context_check_prompt.format(
                context_text=context_text,
                question=user_input,
            )

            # Gemma LLM으로 컨텍스트 체크 (스트리밍 사용)
            llm = GemmaLLM.from_env(project_name="ebs-tutor")
            check_messages = [create_user_message(formatted_check_prompt)]
            check_response = llm.stream(check_messages)

            # 응답 파싱 (스트리밍 응답 처리)
            response_text = ""
            for line in check_response.iter_lines():
                if line:
                    try:
                        line_text = line.decode("utf-8")
                        if line_text.startswith("data: "):
                            line_text = line_text[6:]  # 'data: ' 제거
                            if line_text.strip() == "[DONE]":
                                break
                            try:
                                json_response = json.loads(line_text)
                                if (
                                    json_response.get("choices")
                                    and len(json_response["choices"]) > 0
                                ):
                                    delta = json_response["choices"][0].get("delta", {})
                                    if delta.get("content"):
                                        response_text += delta["content"]
                            except json.JSONDecodeError:
                                continue
                    except UnicodeDecodeError:
                        continue

            # 응답 파싱
            check_lines = response_text.strip().split("\n", 1)
            can_answer = check_lines[0].strip() == "Yes"
            processed_prompt = user_input  # 항상 원래 질문 사용

            # can_answer 값을 세션 상태에 저장
            if "can_answer" not in st.session_state:
                st.session_state["can_answer"] = {}
            st.session_state["can_answer"][question_id] = can_answer

        # 2단계: 최종 응답 생성
        with st.chat_message("assistant"):
            response_container = st.empty()
            context_container = st.empty()  # 컨텍스트 표시용 컨테이너

            try:
                # 튜터 응답 프롬프트 로드 (교재 내용 유무에 따라 다른 프롬프트 사용)
                prompt_name = "context_based_answer" if can_answer else "general_answer"
                tutor_prompt = load_prompt(
                    "prompts/ebs_tutor.yaml", prompt_name=prompt_name
                )

                # 채팅 기록 구성 (최근 3회차 대화 포함)
                chat_history = "\n".join(
                    [
                        f"{msg.role}: {msg.content}"
                        for msg in st.session_state["messages"][-6:-1]
                    ]
                )

                # 튜터 응답 프롬프트 포맷
                formatted_tutor_prompt = tutor_prompt.format(
                    context_text=context_text if can_answer else "",
                    question=processed_prompt,
                    chat_history=chat_history if can_answer else "",
                )

                # Gemma LLM으로 최종 응답 생성
                messages = [create_user_message(formatted_tutor_prompt)]
                response = llm.stream(messages)

                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            line_text = line.decode("utf-8")
                            if line_text.startswith("data: "):
                                line_text = line_text[6:]  # 'data: ' 제거
                                if line_text.strip() == "[DONE]":
                                    break  # 스트리밍 완료
                                try:
                                    json_response = json.loads(line_text)
                                    if (
                                        json_response.get("choices")
                                        and len(json_response["choices"]) > 0
                                    ):
                                        delta = json_response["choices"][0].get(
                                            "delta", {}
                                        )
                                        if delta.get("content"):
                                            full_response += delta["content"]
                                            # 답변이 시작되면 컨텍스트 컨테이너를 비움
                                            if full_response.strip():
                                                context_container.empty()
                                            response_container.markdown(
                                                full_response + "▌"
                                            )
                                except json.JSONDecodeError:
                                    continue  # JSON 파싱 실패한 라인은 무시
                        except UnicodeDecodeError:
                            continue  # 디코딩 실패한 라인은 무시

                # 최종 응답 표시
                if full_response:  # 응답이 있는 경우에만 표시
                    response_container.markdown(full_response)
                    response_text = full_response
                    # 교재에서 답을 찾을 수 없는 경우 안내 메시지만 표시
                    if not can_answer:
                        sources = [
                            {
                                "message": CONTENT_NOT_IN_TEXTBOOK,
                            }
                        ]
                else:
                    response_text = (
                        "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
                    )
                    sources = []

            except Exception as e:
                st.error(f"Gemma3 API 호출 중 오류 발생: {str(e)}")
                response_text = "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
                sources = []

        # 응답 및 소스 저장
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response_text)
        )
        st.session_state["sources"][question_id] = sources

        # PDF 뷰어 페이지 업데이트 (교재에서 답을 찾은 경우에만)
        if can_answer and sources:
            try:
                first_source = sources[0]
                if isinstance(first_source, str):
                    page_no = int(first_source.replace("페이지", ""))
                    st.session_state["pdf_viewer_state"]["current_page"] = page_no
            except (ValueError, IndexError):
                pass  # 페이지 번호 파싱 실패 시 무시

    except Exception as e:
        st.error(f"검색 중 오류 발생: {str(e)}")
