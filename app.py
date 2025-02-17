import streamlit as st

from constants import GREETING_MESSAGE
from custom_logging import langsmith
from services.load_or_create_index import load_or_create_index
from services.search import search_top_k
from utils.chat import (
    add_message,
    create_chain,
    detect_language,
    get_context_text,
    is_greeting,
    print_messages,
    rewrite_query,
    summarize_sources,
    translate_text,
)
from utils.logging import log_debug

langsmith(project_name="Yong-in RAG")

INDEX_FILE = "faiss_index.bin"
CHUNKED_FILE = "chunked_data.pkl"
DATA_DIR = "data/yongin_data2"
MODELS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4-turbo": "gpt-4-turbo",
}

load_or_create_index()

##############################
## 타이틀 및 인사말 추가
##############################
st.title("용인 시청 RAG 챗봇")
st.write(
    "안녕하세요! 용인시 관련 정보를 알고 싶으시면 아래 채팅창에 질문을 입력해 주세요."
)

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    chain = create_chain(model_name=MODELS["gpt-4o-mini"])
    st.session_state["chain"] = chain

# 최초 접속 시 챗봇 인사말 자동 추가
if not st.session_state["messages"]:
    add_message("assistant", GREETING_MESSAGE)

selected_model = MODELS["gpt-4o-mini"]

print_messages()

# 사용자 입력 처리 (챗 입력)
user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning_msg = st.empty()

if user_input:
    chain = st.session_state["chain"]
    if chain is not None:
        # 단순 인사말이면 인사말 응답 처리
        if is_greeting(user_input):
            assistant_reply = (
                "안녕하세요! 용인시청 챗봇입니다. 궁금하신 사항이 있으시면 편하게 말씀해 주세요.\n\n"
                "예시 질문:\n"
                "- 용인시청 전화번호 알려줘\n"
                "- 대형 생활폐기물 어떻게 버려?\n"
                "- 일반 쓰레기 동별 배출일 알려주세요\n"
                "- 여권발급 필요서류 및 발급기간은?\n"
                "- 용인시 공원 예약은 어떻게 하나요?\n"
                "- 용인시 시내버스 노선 알려줘"
            )
            st.chat_message("assistant").write(assistant_reply)
            add_message("assistant", assistant_reply)
        else:
            st.chat_message("user").write(user_input)

            # 1차 검색: 사용자 입력 그대로 사용
            query_for_search = user_input
            results = search_top_k(query_for_search, top_k=3, ranking_mode="rrf")
            log_debug(f"1차 검색 쿼리 = {query_for_search}")
            log_debug(f"1차 RAG 결과 = {results}")

            # 2차 검색: 결과 없으면 재작성 쿼리 사용
            if not results or len(results) == 0:
                with st.spinner("검색 쿼리 재작성 중입니다..."):
                    query_for_search = rewrite_query(user_input)
                results = search_top_k(query_for_search, top_k=3, ranking_mode="rrf")
                log_debug(f"2차 검색 쿼리 = {query_for_search}")
                log_debug(f"2차 RAG 결과 = {results}")

            # RAG 결과 평가 및 fallback
            def get_context_text(results):
                if results and len(results) > 0:
                    summarized = summarize_sources(results)
                    if len(summarized) < 50 or "내용 없음" in summarized:
                        return None
                    return f"📌 **출처 기반 정보**\n{summarized}"
                return None

            context_text = get_context_text(results)
            log_debug(f"최종 context_text = {context_text}")
            if context_text is None:
                context_text = (
                    "📌 **AI 생성 답변**\n검색된 공식 문서가 부족합니다. 아래 답변은 자동 생성된 것입니다. "
                    "이 답변은 부정확할 수 있으므로 반드시 공식 홈페이지(yongin.go.kr)를 확인해 주세요."
                )

            # 이전 대화 내역은 포함하지 않고, 오직 현재 질문만을 사용합니다.
            combined_query = (
                f"아래는 관련 문서 내용 (RAG):\n{context_text}\n\n"
                f"최종 질문: {user_input}"
            )
            log_debug(f"최종 쿼리 = {combined_query}")

            # 다국어 처리: 입력 언어 감지 후 필요 시 번역
            detected_lang = detect_language(user_input)
            log_debug(f"감지된 언어 = {detected_lang}")
            if detected_lang != "ko":
                combined_query = translate_text(combined_query, detected_lang)
                log_debug(f"번역된 최종 쿼리 = {combined_query}")

            # 스트리밍 응답 생성
            response_generator = chain.stream(combined_query)
            with st.chat_message("assistant"):
                container = st.empty()
                ai_answer = ""
                spinner_placeholder = st.empty()
                spinner_placeholder.markdown("**답변 생성 중입니다...**")
                for token in response_generator:
                    if ai_answer == "":
                        spinner_placeholder.empty()
                    ai_answer += token
                    container.markdown(ai_answer)
            log_debug(f"최종 AI 답변 = {ai_answer}")

            # 대화 기록 저장
            add_message("user", user_input)
            add_message("assistant", ai_answer)
    else:
        warning_msg.error("체인을 생성할 수 없습니다.")
