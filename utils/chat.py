import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from utils.custom_logging import gemma_trace
from utils.prompts import load_prompt

MODELS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4-turbo": "gpt-4-turbo",
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


def create_chain(model_name=MODELS["gpt-4o-mini"]):
    """
    프롬프트를 로드한 후, 스트리밍 응답 체인을 반환합니다.
    """
    prompt = load_prompt("prompts/yongin.yaml")
    llm = ChatOpenAI(model_name=model_name, temperature=0.8, streaming=True)
    chain = {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return chain


def rewrite_query(user_question: str) -> str:
    """
    사용자의 질문을 분석하여, 용인시청 관련 최신 정보를 효과적으로 검색할 수 있는 최적화된 검색 쿼리를 생성합니다.
    """
    llm_rewriter = ChatOpenAI(
        model_name=MODELS["gpt-4o-mini"], temperature=0.8, streaming=False
    )
    rewriter_prompt = (
        "다음 질문을 분석하여 용인시청 관련 최신 정보(정책, 서비스, 행사 등)를 효과적으로 검색할 수 있는 "
        "핵심 키워드와 문장을 포함한 최적의 검색 쿼리를 만들어주세요. 답변은 간결하고 구체적으로 작성해 주세요.\n"
        f"질문: {user_question}\n"
        "최적화된 검색 쿼리:"
    )
    rewritten = llm_rewriter.invoke(rewriter_prompt)
    return (
        rewritten.content.strip()
        if hasattr(rewritten, "content")
        else str(rewritten).strip()
    )


def summarize_conversation(history_text: str) -> str:
    """
    주어진 대화 내역을 간단하게 요약하여 반환합니다.
    """
    llm_summary = ChatOpenAI(
        model_name=MODELS["gpt-4o-mini"], temperature=0, streaming=False
    )
    summary_prompt = (
        f"다음 대화 내용을 간단하게 요약해줘:\n\n{history_text}\n\n간단하게 요약해줘."
    )
    summary_response = llm_summary.invoke(summary_prompt)
    if hasattr(summary_response, "content"):
        return summary_response.content.strip()
    else:
        return str(summary_response).strip()


def detect_language(text: str) -> str:
    """
    주어진 텍스트의 전체 맥락을 고려하여, 해당 언어의 ISO 639-1 코드를 반환합니다.
    예: 한국어 'ko', 영어 'en', 일본어 'ja'
    """
    llm = ChatOpenAI(model_name=MODELS["gpt-4o-mini"], temperature=0, streaming=False)
    prompt = (
        "다음 텍스트의 전체 내용을 고려하여, 해당 텍스트가 어떤 언어로 작성되었는지 ISO 639-1 코드로 한 단어로 응답해 주세요. "
        "예시: 'ko' (한국어), 'en' (영어), 'ja' (일본어).\n"
        f"텍스트: '''{text}'''"
    )
    response = llm.invoke(prompt)
    lang = response.content.strip() if hasattr(response, "content") else "ko"
    if lang not in {"ko", "en", "ja"}:
        lang = "ko"
    return lang


def translate_text(text: str, target_lang: str) -> str:
    """
    주어진 텍스트를 target_lang 언어로 번역합니다.
    """
    llm = ChatOpenAI(model_name=MODELS["gpt-4o-mini"], temperature=0, streaming=False)
    prompt = f"다음 텍스트를 '{target_lang}'로 번역해줘:\n{text}"
    response = llm.invoke(prompt)
    return response.content.strip() if hasattr(response, "content") else text


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
