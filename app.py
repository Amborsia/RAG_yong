# app.py
import os
import pickle

import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI

import models.database as db
from services.initialize import init_rag
from services.load_or_create_index import load_or_create_index
from services.search import search_top_k
from utils.chat import (
    add_message,
    create_chain,
    get_context_text,
    print_messages,
    rewrite_query,
    summarize_sources,
)
from utils.custom_logging import langsmith
from utils.greeting_message import GREETING_MESSAGE
from utils.logging import log_debug

langsmith(project_name="Yong-in RAG")

# 채팅 입력창 높이 조정을 위한 CSS 추가
st.markdown(
    """
<style>
.stMain {
    position: relative;
}
.stChatMessage {
    background-color: transparent !important;
}
[data-testid=stSidebar] {
    background-color: #3d9df3;
    padding:0 15px;
}

[data-testid=stSidebarUserContent] {
    background-color: white;
    border-radius: 10px;
}

/* 채팅 입력창 높이 조정 - 새로운 클래스명 사용 */
.st-emotion-cache-glsyku {
    min-height: 80px !important;
    align-items: center;
}
.st-emotion-cache-glsyku textarea:active{
    outline: none;
}
.st-emotion-cache-glsyku div {
    display: flex;
    align-items: center;
    justify-content: center;
}

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
[data-baseweb="textarea"] {
    border-color: transparent !important;
}
/* 채팅 입력창 높이 조정 */
/*
.st-emotion-cache-qcqlej {
    height: 0 !important;
    flex-grow: 0 !important;
}
*/
""",
    unsafe_allow_html=True,
)

# RAG 모드 설정
RAG_MODES = {
    "base": {
        "name": "기본 모드",
        "description": "홈페이지 기반 구정 정보, 주요 행사 등을 안내합니다.",
        "index_file": "faiss_index.bin",
        "chunked_file": "chunked_data.pkl",
        "data_dir": "data/yongin_data2",
        "prompt_file": "prompts/yongin_base.yaml",
    },
    "simple": {
        "name": "간단 모드",
        "description": "홈페이지 기반 구정 정보, 주요 행사 등을 간단하게 안내합니다.",
        "index_file": "faiss_index.bin",
        "chunked_file": "chunked_data.pkl",
        "data_dir": "data/yongin_data2",
        "prompt_file": "prompts/yongin_simple.yaml",
    },
    "contact": {
        "name": "조직도 모드",
        "description": "조직도 정보를 기반으로 안내합니다.",
        "index_file": "rag_index/index.faiss",
        "chunked_file": "rag_index/index.pkl",
        "data_dir": "crawling/output",
        "prompt_file": "prompts/yongin_contact.yaml",
    },
    "article": {
        "name": "기사 작성 모드",
        "description": "기사를 작성합니다.",
        "prompt_file": "prompts/yongin_article.yaml",
    },
    "research": {  # 🆕 이력서 작성 모드 추가
        "name": "연구과제작성 모드",
        "description": "연구과제를 작성합니다.",
        "prompt_file": "prompts/yongin_research.yaml",
    },
    "policy": {  # 🆕 이메일 작성 모드 추가
        "name": "정책보고서 모드",
        "description": "정책보고서를 작성합니다.",
        "prompt_file": "prompts/yongin_policy.yaml",
    },
    "event_doc": {  # 🆕 제안서 작성 모드 추가
        "name": "행사보고서 모드",
        "description": "행사 보고서를 작성합니다.",
        "prompt_file": "prompts/yongin_event_doc.yaml",
    },
}

# 전역 변수 제거 (모드별 설정으로 대체)
# INDEX_FILE = "rag_index/index.faiss"
# CHUNKED_FILE = "rag_index/index.pkl"
# DATA_DIR = "crawling/output"


# --- 추가: 대화 내역 요약 함수 ---
def summarize_conversation(history_text: str) -> str:
    """
    주어진 대화 내역을 간단하게 요약하여 반환합니다.
    """
    llm_summary = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=False)
    summary_prompt = (
        f"다음 대화 내용을 간단하게 요약해줘:\n\n{history_text}\n\n간단하게 요약해줘."
    )
    summary_response = llm_summary.invoke(summary_prompt)
    if hasattr(summary_response, "content"):
        summary_text = summary_response.content
    else:
        summary_text = str(summary_response)
    return summary_text.strip()


##############################
## FAISS 인덱스 자동 생성 및 로드
##############################


def reset_db_state():
    """
    데이터베이스 상태를 초기화합니다.
    모드 전환 시 이전 모드의 데이터가 남아있는 것을 방지합니다.
    """
    # 기존 데이터 초기화
    db.documents = []
    db.chunked_data = {}
    db.index = None


def load_or_create_index(mode="base"):
    """
    선택된 모드에 따라 FAISS 인덱스를 로드하거나 생성합니다.
    """
    # 데이터베이스 상태 초기화
    reset_db_state()

    mode_config = RAG_MODES[mode]

    # 📌 "doc" 모드는 RAG를 사용하지 않으므로 인덱스를 로드할 필요 없음
    if mode in ["article", "research", "policy", "event_doc"]:
        log_debug("📌 'doc' 모드에서는 FAISS 인덱스를 로드하지 않습니다.")
        return

    # 일반적인 RAG 모드 처리
    INDEX_FILE = mode_config.get("index_file", None)
    CHUNKED_FILE = mode_config.get("chunked_file", None)
    DATA_DIR = mode_config.get("data_dir", None)

    if not INDEX_FILE or not CHUNKED_FILE or not DATA_DIR:
        log_debug(f"❌ {mode} 모드에서 필요한 파일 설정이 없습니다.")
        return

    if os.path.exists(INDEX_FILE):
        db.load_data(DATA_DIR)
        db.load_index(INDEX_FILE, index_type="FLAT")

        try:
            with open(CHUNKED_FILE, "rb") as f:
                loaded_chunked = pickle.load(f)
            db.chunked_data = (
                loaded_chunked[0]
                if isinstance(loaded_chunked, tuple)
                else loaded_chunked
            )

            log_debug(f"문서 개수: {len(db.documents)}")
            log_debug(f"청크 개수: {len(db.chunked_data.get('all_chunks', []))}")
            log_debug(f"인덱스 크기: {db.index.ntotal if db.index else 0}")

        except FileNotFoundError:
            st.warning(f"⚠️ `{CHUNKED_FILE}` 파일이 없습니다.")
        except Exception as e:
            st.error(f"❌ `{CHUNKED_FILE}` 로드 중 오류 발생: {e}")

    else:
        st.write(f"🔄 FAISS 인덱스({INDEX_FILE})가 없습니다. 새로 생성 중...")
        init_rag(
            data_dir=DATA_DIR,
            chunk_strategy="recursive",
            chunk_param=500,
            index_type="FLAT",
            output_index_path=INDEX_FILE,
            output_chunk_path=CHUNKED_FILE,
        )
        st.success("✅ 새로운 FAISS 인덱스 생성 완료!")
        db.load_data(DATA_DIR)
        db.load_index(INDEX_FILE, index_type="FLAT")

        try:
            with open(CHUNKED_FILE, "rb") as f:
                loaded_chunked = pickle.load(f)
            db.chunked_data = (
                loaded_chunked[0]
                if isinstance(loaded_chunked, tuple)
                else loaded_chunked
            )
            st.success("✅ 인덱스 및 chunked_data 로드 완료!")
        except FileNotFoundError:
            st.warning(f"⚠️ `{CHUNKED_FILE}` 파일이 여전히 없습니다.")


##############################
## 타이틀 및 인사말 추가
##############################
# 타이틀은 사이드바에만 표시
with st.sidebar:
    st.title("용인 시청 RAG 챗봇")

# 사이드바: 모드 선택 및 초기화 버튼
with st.sidebar:
    st.title("RAG 모드 설정")

    # 세션 상태에 모드 저장
    if "rag_mode" not in st.session_state:
        st.session_state["rag_mode"] = "base"

    # 대화 시작 여부 추적
    if "conversation_started" not in st.session_state:
        st.session_state["conversation_started"] = False

    # 모드 선택 라디오 버튼
    selected_mode = st.radio(
        "모드 선택",
        options=list(RAG_MODES.keys()),
        format_func=lambda x: f"{RAG_MODES[x]['name']}",
        index=list(RAG_MODES.keys()).index(st.session_state["rag_mode"]),
    )

    # 모드가 변경되었을 때
    if selected_mode != st.session_state["rag_mode"]:
        st.session_state["rag_mode"] = selected_mode
        # 대화 내역 초기화
        if "messages" in st.session_state:
            st.session_state["messages"] = []
        # 체인 초기화 (새 모드의 프롬프트로 다시 생성)
        if "chain" in st.session_state:
            st.session_state.pop("chain")
        # 새 모드로 인덱스 로드
        load_or_create_index(selected_mode)
        st.success(f"✅ {RAG_MODES[selected_mode]['name']}로 전환되었습니다!")
        # 페이지 새로고침 (모드 변경 적용을 위해)
        st.rerun()

    # 디버그 정보 표시
    # with st.expander("디버그 정보"):
    #     st.write(f"현재 모드: {st.session_state['rag_mode']}")
    #     st.write(f"문서 개수: {len(db.documents)}")
    #     st.write(f"청크 개수: {len(db.chunked_data.get('all_chunks', []))}")
    #     st.write(f"인덱스 크기: {db.index.ntotal if db.index else 0}")

    #     if st.button("데이터 다시 로드"):
    #         reset_db_state()
    #         load_or_create_index(st.session_state["rag_mode"])
    #         st.success("✅ 데이터 다시 로드 완료!")
    #         st.rerun()

# 현재 모드로 인덱스 로드
load_or_create_index(st.session_state["rag_mode"])


# 이전 대화 기록 출력 및 메시지 추가 함수
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# def is_greeting(text: str) -> bool:
#     """
#     인사말 여부를 판단하는 함수.
#     간단한 인사말은 환영 메시지로 처리합니다.
#     """
#     greetings = ["안녕", "안녕?", "안녕하세요", "안녕하세요?"]
#     return text.strip() in greetings


# --- 커스텀 프롬프트 runnable 정의 ---
# 프롬프트 템플릿 파일을 읽어서 사용자 질문을 채워 넣는 역할
def load_prompt(file_path: str, encoding: str = "utf-8") -> str:
    with open(file_path, encoding=encoding) as f:
        return f.read()


class RunnablePrompt(Runnable):
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template

    def invoke(self, input_dict: dict, config=None, **kwargs) -> str:
        # input_dict에서 "question" 키를 가져와 템플릿에 채워 넣습니다.
        question = input_dict.get("question", "")
        prompt_text = self.prompt_template.format(question=question)
        # 이제 문자열(prompt_text)만 반환합니다.
        return prompt_text

    # (추가 메서드 구현은 필요에 따라)


# --- 체인 생성 ---
def create_chain(model_name="gpt-4o", mode="base"):
    # 현재 모드에 맞는 프롬프트 파일 로드
    prompt_file = RAG_MODES[mode]["prompt_file"]
    print(prompt_file)
    prompt_template = load_prompt(prompt_file, encoding="utf-8")
    # 커스텀 프롬프트 runnable 생성
    prompt_runnable = RunnablePrompt(prompt_template)
    # streaming=True 옵션을 추가하여 스트리밍 응답을 활성화합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0, streaming=True)
    # 체인 구성: 초기 입력은 {"question": <사용자 질문>}를 RunnablePassthrough()로 그대로 넘기고,
    # 그 다음 커스텀 프롬프트 runnable로 템플릿 적용, 이후 llm 호출, 마지막에 StrOutputParser()로 결과 파싱.
    chain = (
        {"question": RunnablePassthrough()} | prompt_runnable | llm | StrOutputParser()
    )
    return chain


# 검색 쿼리 재작성
# def rewrite_query(user_question: str) -> str:
#     """
#     LLM을 활용하여 사용자 질문을 바탕으로 용인시청 관련 최신 정보를 포함할 수 있는 검색 쿼리를 동적으로 재작성합니다.
#     """
#     llm_rewriter = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=False)
#     rewriter_prompt = (
#         "다음 질문을 바탕으로 용인시청에 관련된 최신 정보를 검색하기 위한 최적의 검색 쿼리를 만들어줘. "
#         "질문의 의미와 관련 키워드를 고려해서 검색 결과에 최신 소식이 잘 포함될 수 있도록 작성해줘.\n"
#         f"질문: {user_question}\n"
#         "검색 쿼리:"
#     )
#     rewritten = llm_rewriter.invoke(rewriter_prompt)
#     if hasattr(rewritten, "content"):
#         return rewritten.content.strip()
#     else:
#         return str(rewritten).strip()


# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    current_mode = st.session_state["rag_mode"]
    chain = create_chain(model_name="gpt-4o", mode=current_mode)
    st.session_state["chain"] = chain

# 최초 접속 시 챗봇 인사말 자동 추가
if not st.session_state["messages"]:
    current_mode = st.session_state["rag_mode"]
    add_message("assistant", GREETING_MESSAGE[current_mode])

print_messages()
# 채팅 입력창 추가
user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning_msg = st.empty()

if user_input:

    if st.session_state["rag_mode"] in ["article", "research", "policy", "event_doc"]:
        st.chat_message("user").write(user_input)
        # ✅ "doc" 모드에서도 프롬프트를 적용
        prompt_file = RAG_MODES[st.session_state["rag_mode"]]["prompt_file"]
        prompt_template = load_prompt(prompt_file, encoding="utf-8")

        # ✅ 사용자 입력을 프롬프트에 적용 (question 변수로 전달)
        formatted_query = prompt_template.format(question=user_input)

        # ✅ GPT 호출 (gpt-4o 사용)
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)
        response_generator = llm.stream(formatted_query)

        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            spinner_placeholder = st.empty()
            spinner_placeholder.markdown("**답변 생성 중입니다...**")

            for token in response_generator:
                if hasattr(token, "content"):  # ✅ AIMessageChunk 객체일 경우
                    token_text = token.content
                else:
                    token_text = str(token)  # ✅ 문자열 변환

                if ai_answer == "":
                    spinner_placeholder.empty()

                ai_answer += token_text
                container.markdown(ai_answer)

        log_debug(f"최종 AI 답변 (doc 모드) = {ai_answer}")

        add_message("user", user_input)
        add_message("assistant", ai_answer)

    else:
        # ✅ "doc" 모드가 아닌 경우, 기존 RAG 검색 수행
        chain = st.session_state["chain"]
        if chain is not None:
            st.chat_message("user").write(user_input)
            try:
                query_for_search = user_input
                results = search_top_k(query_for_search, top_k=5, ranking_mode="rrf")
                # log_debug(f"1차 검색 결과 개수: {len(results)}")

                # if not results:
                #     with st.spinner(""):
                #         query_for_search = rewrite_query(user_input)
                #     results = search_top_k(
                #         query_for_search, top_k=3, ranking_mode="rrf"
                #     )
                #     log_debug(f"2차 검색 결과 개수: {len(results)}")
            except Exception as e:
                log_debug(f"검색 중 오류 발생: {str(e)}")
                results = []
                st.error(f"검색 중 오류가 발생했습니다: {str(e)}")

            # RAG 결과 평가 및 fallback
            def get_context_text(results):
                if results and len(results) > 0:
                    summarized = summarize_sources(results)
                    if len(summarized) < 50 or "내용 없음" in summarized:
                        return None
                    return f"📌 **출처 기반 정보**\n{summarized}"
                return None

            # 검색 결과가 있는 경우에만 처리
            if results and len(results) > 0:
                context_text = get_context_text(results)
                log_debug(f"최종 context_text = {context_text}")

                answer_chunks = []
                for r in results[:3]:
                    chunk_text = r.get("chunk_text", "내용 없음")
                    doc_url = r.get("original_doc", {}).get("url", "출처 없음")
                    enriched_chunk = (
                        f"이 chunk는 {doc_url} 에서 가져온 내용입니다.\n{chunk_text}"
                    )
                    answer_chunks.append(enriched_chunk)
                context_text = "\n\n".join(answer_chunks)
            else:
                # 검색 결과가 없는 경우 기본 메시지 사용
                context_text = (
                    get_context_text(results)
                    if results
                    else "📌 **AI 생성 답변**\n검색된 공식 문서가 부족합니다. 아래 답변은 자동 생성된 것입니다. "
                    "이 답변은 부정확할 수 있으므로 반드시 공식 홈페이지(yongin.go.kr)를 확인해 주세요."
                )

            # conversation_history = ""
            # if len(st.session_state["messages"]) > 1:
            #     recent_msgs = st.session_state["messages"][-5:]
            #     for msg in recent_msgs:
            #         conversation_history += f"{msg.role.capitalize()}: {msg.content}\n"
            #     if len(conversation_history) > 800:
            #         conversation_history = summarize_conversation(conversation_history)

            # conversation_section = (
            #     f"이전 대화 내용:\n{conversation_history}\n"
            #     if conversation_history
            #     else ""
            # )
            combined_query = (
                f"관련 문서 내용 (RAG):{context_text}\n\n"
                f"최종 질문: {user_input}"
                # f"{conversation_section}최종 질문: {user_input}"
            )

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

            log_debug(f"최종 AI 답변 (한국어) = {ai_answer}")

            add_message("user", user_input)
            add_message("assistant", ai_answer)

        else:
            st.error("체인을 생성할 수 없습니다.")
