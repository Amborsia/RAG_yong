import json
import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage

from services.ebs import EbsRAG
from utils.prompts import load_prompt

from .search import process_search_results

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
            results = ebs_rag.search(user_input, top_k=3)
            st.session_state["search_results"] = results
            st.session_state["question_results"][question_id] = results

            context_text, sources = process_search_results(results, question_id)

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

        # Gemma3 API 호출
        with st.chat_message("assistant"):
            response_container = st.empty()
            try:
                response = requests.post(
                    f"{GEMMA_URL}/v1/chat/completions",
                    json={
                        "model": "chat_model",
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": formatted_prompt}],
                            }
                        ],
                        "max_tokens": 1000,
                        "stream": True,
                    },
                    headers={
                        "Authorization": f"Bearer {os.getenv('GEMMA_TOKEN')}",
                        "Content-Type": "application/json",
                    },
                    stream=True,
                )
                response.raise_for_status()

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
                else:
                    response_text = (
                        "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
                    )
            except Exception as e:
                st.error(f"Gemma3 API 호출 중 오류 발생: {str(e)}")
                response_text = "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."

        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response_text)
        )
        st.session_state["sources"][question_id] = sources
        if results:
            st.session_state["pdf_viewer_state"]["current_page"] = results[0]["page_no"]

    except Exception as e:
        st.error(f"검색 중 오류 발생: {str(e)}")
