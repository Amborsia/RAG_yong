import os
import pickle

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import models.database as db
from custom_logging import langsmith
from initialize import init_rag
from prompts import load_prompt
from services.search import search_top_k

langsmith(project_name="Yong-in RAG")

# ✅ FAISS 인덱스 파일 경로
INDEX_FILE = "faiss_index.bin"
CHUNKED_FILE = "chunked_data.pkl"
DATA_DIR = "data/yongin_data2"
MODELS = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4-turbo": "gpt-4-turbo",
}


# --- 추가: 대화 내역 요약 함수 ---
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
    # 추천: invoke 메서드를 사용하여 호출 (또는 __call__ 결과에서 content 추출)
    summary_response = llm_summary.invoke(summary_prompt)
    # 반환된 결과가 AIMessage 객체라면 content 속성에서 텍스트 추출
    if hasattr(summary_response, "content"):
        summary_text = summary_response.content
    else:
        summary_text = str(summary_response)
    return summary_text.strip()


# FAISS 인덱스 자동 생성 및 로드
def load_or_create_index():
    if os.path.exists(INDEX_FILE):
        # 문서 데이터 로드
        db.load_data(DATA_DIR)

        # FAISS 인덱스 로드
        db.load_index(INDEX_FILE, index_type="FLAT")

        # chunked_data 로드 (없으면 경고)
        try:
            with open(CHUNKED_FILE, "rb") as f:
                db.chunked_data = pickle.load(f)
        except FileNotFoundError:
            st.warning(
                "⚠️ `chunked_data.pkl` 파일이 없습니다. 일부 기능이 제한될 수 있습니다."
            )
        except Exception as e:
            st.error(f"❌ `chunked_data.pkl` 로드 중 오류 발생: {e}")
    else:
        st.write("🔄 FAISS 인덱스가 없습니다. 새로 생성 중...")
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
                db.chunked_data = pickle.load(f)
            st.success("✅ 인덱스 및 chunked_data 로드 완료!")
        except FileNotFoundError:
            st.warning("⚠️ `chunked_data.pkl` 파일이 여전히 없습니다.")


load_or_create_index()

##############################
## 타이틀 및 인사말 추가
##############################
st.title("용인 시청 RAG 챗봇")


# 이전 대화 기록 출력 및 메시지 추가 함수
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def is_greeting(text: str) -> bool:
    """
    인사말 여부를 판단하는 함수.
    간단한 인사말은 환영 메시지로 처리합니다.
    """
    greetings = ["안녕", "안녕?", "안녕하세요", "안녕하세요?"]
    return text.strip() in greetings


# 체인 생성 (LangChain의 스트리밍 모드를 적용)
def create_chain(model_name=MODELS["gpt-4-turbo"]):
    prompt = load_prompt("prompts/yongin.yaml")

    # streaming=True 옵션을 추가하여 스트리밍 응답을 활성화합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0.2, streaming=True)
    chain = {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return chain


# 검색 쿼리 재작성
def rewrite_query(user_question: str) -> str:
    """
    LLM을 활용하여 사용자 질문을 바탕으로 용인시청 관련 최신 정보를 포함할 수 있는 검색 쿼리를 동적으로 재작성합니다.
    """
    llm_rewriter = ChatOpenAI(
        model_name=MODELS["gpt-4o-mini"], temperature=0.3, streaming=False
    )
    rewriter_prompt = (
        "다음 질문을 바탕으로 용인시청에 관련된 최신 정보를 검색하기 위한 최적의 검색 쿼리를 만들어줘. "
        "질문의 의미와 관련 키워드를 고려해서 검색 결과에 최신 소식이 잘 포함될 수 있도록 작성해줘.\n"
        f"질문: {user_question}\n"
        "검색 쿼리:"
    )
    rewritten = llm_rewriter.invoke(rewriter_prompt)
    if hasattr(rewritten, "content"):
        return rewritten.content.strip()
    else:
        return str(rewritten).strip()


# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    chain = create_chain(model_name=MODELS["gpt-4o-mini"])
    st.session_state["chain"] = chain

## 최초 접속 시 챗봇 인사말 자동 추가 (대화가 시작되지 않은 경우)
if not st.session_state["messages"]:
    greeting_msg = "안녕하세요! 용인시청 챗봇입니다. 궁금하신 사항이 있으시면 편하게 말씀해 주세요."
    add_message("assistant", greeting_msg)

# 사이드바: 초기화 버튼과 모델 선택 메뉴
# with st.sidebar:
#     clear_btn = st.button("대화 초기화")
#     selected_model = st.selectbox(
#         "LLM 선택", MODELS.keys(), index=0
#     )
selected_model = MODELS["gpt-4o-mini"]

# if clear_btn:
#     st.session_state["messages"] = []

print_messages()

# 사용자 입력 처리 (챗 입력)
user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning_msg = st.empty()

if user_input:
    chain = st.session_state["chain"]
    if chain is not None:
        # 사용자 입력 출력
        st.chat_message("user").write(user_input)

        # 먼저 사용자 입력을 그대로 검색 쿼리로 사용합니다.
        query_for_search = user_input
        results = search_top_k(query_for_search, top_k=5, ranking_mode="rrf")
        # 만약 검색 결과가 없으면, 사용자 질문을 재작성한 검색 쿼리로 다시 시도합니다.
        if not results or len(results) == 0:
            with st.spinner("검색 쿼리 재작성 중입니다..."):
                query_for_search = rewrite_query(user_input)
            results = search_top_k(query_for_search, top_k=5, ranking_mode="rrf")
        if not results or len(results) == 0:
            context_text = "❌ 관련 문서가 없습니다."
        else:
            answer_chunks = []
            for r in results[:3]:
                chunk_text = r.get("chunk_text", "내용 없음")
                doc_url = r.get("original_doc", {}).get("url", "출처 없음")
                enriched_chunk = (
                    f"이 chunk는 {doc_url} 에서 가져온 내용입니다.\n{chunk_text}"
                )
                answer_chunks.append(enriched_chunk)
            context_text = "\n\n".join(answer_chunks)
        # --- RAG: end ---

        # --- 선택: 대화 기록 구성 ---
        conversation_history = ""
        if st.session_state["messages"]:
            recent_msgs = st.session_state["messages"][-5:]
            for msg in recent_msgs:
                conversation_history += f"{msg.role.capitalize()}: {msg.content}\n"
            if len(conversation_history) > 500:
                conversation_history = summarize_conversation(conversation_history)
        conversation_section = ""
        if conversation_history:
            conversation_section = f"이전 대화 내용:\n{conversation_history}\n"
        # 최종 combined_query 구성 (RAG 내용 + 대화 맥락 + 현재 질문)
        combined_query = (
            f"아래는 관련 문서 내용 (RAG):\n{context_text}\n\n"
            f"{conversation_section}"
            f"최종 질문: {user_input}"
        )

        # chain.stream() 호출은 별도에서 진행하고 spinner를 개별 placeholder로 처리합니다.
        response_generator = chain.stream(combined_query)
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            # spinner를 위한 별도 placeholder를 생성합니다.
            spinner_placeholder = st.empty()
            spinner_placeholder.markdown("**답변 생성 중입니다...**")
            for token in response_generator:
                # 첫 토큰이 도착하면 spinner를 제거합니다.
                if ai_answer == "":
                    spinner_placeholder.empty()
                ai_answer += token
                container.markdown(ai_answer)

        # 대화 기록에 메시지 추가
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_msg.error("체인을 생성할 수 없습니다.")
