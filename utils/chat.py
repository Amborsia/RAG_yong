import json
import os

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_core.runnables import RunnablePassthrough

from utils.custom_logging import gemma_trace
from utils.prompts import load_prompt

# 환경 변수 로드
load_dotenv()

# Gemma3 설정
GEMMA_URL = os.getenv("GEMMA_URL", "http://localhost:8000")  # 기본값 설정
MODELS = {
    "gemma3": "chat_model",  # Gemma3 모델명
}


def print_messages():
    """세션에 저장된 대화 기록을 순차적으로 출력합니다."""
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    """세션에 새 메시지를 추가합니다."""
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def is_greeting(text: str) -> bool:
    """
    단순 인사말(예: "안녕", "안녕하세요")만을 인식하도록 개선합니다.
    """
    return text.strip() in {"안녕", "안녕하세요"}


def filter_conversation(history_msgs):
    """
    대화 내역에서 인사말, TIP, 예시 질문 등 불필요한 부분을 제거합니다.
    """
    filtered = []
    exclusion_keywords = {
        "안녕",
        "안녕하세요",
        "TIP!",
        "예시 질문",
        "더 나은 삶을 위한 스마트도시",
    }
    for msg in history_msgs:
        if not any(msg.content.startswith(keyword) for keyword in exclusion_keywords):
            filtered.append(msg)
    return filtered


@gemma_trace(project_name="ebs-science")
def gemma_call(prompt: str) -> str:
    try:
        response = requests.post(
            f"{GEMMA_URL}/v1/chat/completions",
            json={
                "model": MODELS["gemma3"],
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
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
        response_container = st.empty()

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
                                delta = json_response["choices"][0].get("delta", {})
                                if delta.get("content"):
                                    full_response += delta["content"]
                                    response_container.markdown(full_response + "▌")
                        except json.JSONDecodeError:
                            continue  # JSON 파싱 실패한 라인은 무시
                except UnicodeDecodeError:
                    continue  # 디코딩 실패한 라인은 무시

        response_container.markdown(full_response)
        return full_response

    except Exception as e:
        st.error(f"Gemma3 API 호출 중 오류 발생: {str(e)}")
        raise  # 예외를 다시 발생시켜 트레이싱에서 캡처할 수 있도록 함


def create_chain(model_name=MODELS["gemma3"]):
    """
    프롬프트를 로드한 후, Gemma3 API를 사용하는 체인을 반환합니다.
    """
    prompt = load_prompt("prompts/yongin.yaml")
    chain = {"question": RunnablePassthrough()} | prompt | gemma_call
    return chain


def rewrite_query(user_question: str) -> str:
    """
    사용자의 질문을 분석하여, 용인시청 관련 최신 정보를 효과적으로 검색할 수 있는 최적화된 검색 쿼리를 생성합니다.
    """
    rewriter_prompt = (
        "다음 질문을 분석하여 용인시청 관련 최신 정보(정책, 서비스, 행사 등)를 효과적으로 검색할 수 있는 "
        "핵심 키워드와 문장을 포함한 최적의 검색 쿼리를 만들어주세요. 답변은 간결하고 구체적으로 작성해 주세요.\n"
        f"질문: {user_question}\n"
        "최적화된 검색 쿼리:"
    )
    rewritten = gemma_call(rewriter_prompt)
    return rewritten.strip()


def summarize_conversation(history_text: str) -> str:
    """
    주어진 대화 내역을 간단하게 요약하여 반환합니다.
    """
    summary_prompt = (
        f"다음 대화 내용을 간단하게 요약해줘:\n\n{history_text}\n\n간단하게 요약해줘."
    )
    return gemma_call(summary_prompt).strip()


def detect_language(text: str) -> str:
    """
    주어진 텍스트의 전체 맥락을 고려하여, 해당 언어의 ISO 639-1 코드를 반환합니다.
    """
    prompt = (
        "다음 텍스트의 전체 내용을 고려하여, 해당 텍스트가 어떤 언어로 작성되었는지 ISO 639-1 코드로 한 단어로 응답해 주세요. "
        "예시: 'ko' (한국어), 'en' (영어), 'ja' (일본어).\n"
        f"텍스트: '''{text}'''"
    )
    lang = gemma_call(prompt).strip()
    if lang not in {"ko", "en", "ja"}:
        lang = "ko"
    return lang


def translate_text(text: str, target_lang: str) -> str:
    """
    주어진 텍스트를 target_lang 언어로 번역합니다.
    """
    prompt = f"다음 텍스트를 '{target_lang}'로 번역해줘:\n{text}"
    return gemma_call(prompt).strip()


def summarize_sources(results):
    """
    검색된 결과 중 최대 3개의 출처 기반 정보를 요약하여 반환합니다.
    """
    summarized_text = []
    for r in results[:3]:
        chunk_text = r.get("chunk_text", "내용 없음")
        doc_url = r.get("original_doc", {}).get("url", "출처 없음")
        summarized_text.append(f"- {chunk_text} (출처: {doc_url})")
    return "\n".join(summarized_text)


def get_context_text(results):
    """
    검색 결과가 충분한 정보를 제공하는지 평가하여,
    충분하면 출처 기반 정보를 반환하고, 그렇지 않으면 None을 반환합니다.
    """
    if results and len(results) > 0:
        summarized = summarize_sources(results)
        # 예시: 요약된 결과가 50자 미만이거나 "내용 없음"이 포함되면 부족하다고 판단
        if len(summarized) < 50 or "내용 없음" in summarized:
            return None
        return f"📌 **출처 기반 정보**\n{summarized}"
    return None
