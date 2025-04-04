import json
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage

from services.constants import not_found_in
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
        with st.spinner("검색 중..."):
            # 검색과 결과 가공을 한번에 처리
            context_text, sources, results = ebs_rag.search_with_processed_results(
                user_input, question_id, top_k=3
            )

            # 검색 결과 저장
            st.session_state["search_results"] = results
            st.session_state["question_results"][question_id] = results

            # 프롬프트 템플릿 로드
            prompt_template = load_prompt("prompts/ebs_tutor.yaml")

            # 채팅 기록 구성 (최근 3회차 대화 포함)
            chat_history = "\n".join(
                [
                    f"{msg.role}: {msg.content}"
                    for msg in st.session_state["messages"][-6:-1]
                ]
            )

            formatted_prompt = prompt_template.format(
                context_text=context_text,
                question=user_input,
                chat_history=chat_history,
            )

        # Gemma LLM 인스턴스 생성 및 응답 생성
        with st.chat_message("assistant"):
            response_container = st.empty()
            try:
                llm = GemmaLLM.from_env(project_name="ebs-tutor")
                messages = [create_user_message(formatted_prompt)]
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

                    # 응답에 "찾을 수 없는 내용이에요" 포함 여부 확인
                    if not_found_in(response_text):
                        sources = [{"message": "생성된 답변입니다"}]
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

        # PDF 뷰어 페이지 업데이트
        if sources:
            try:
                first_source = sources[0]
                if isinstance(first_source, str):
                    page_no = int(first_source.replace("페이지", ""))
                    st.session_state["pdf_viewer_state"]["current_page"] = page_no
            except (ValueError, IndexError):
                pass  # 페이지 번호 파싱 실패 시 무시

    except Exception as e:
        st.error(f"검색 중 오류 발생: {str(e)}")
