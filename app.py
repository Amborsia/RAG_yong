import os
import pickle

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote import logging
from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter

import models.database as db
from initialize import init_rag

# ✅ FAISS 인덱스 파일 경로
INDEX_FILE = "faiss_index.bin"
CHUNKED_FILE = "chunked_data.pkl"
DATA_DIR = "data/yongin_data2"


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
st.write(
    "안녕하세요! 용인시 관련 정보를 알고 싶으시면 아래 채팅창에 질문을 입력해 주세요."
)


# 이전 대화 기록 출력 및 메시지 추가 함수
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인 생성 (LangChain의 스트리밍 모드를 적용)
def create_chain(model_name="gpt-4o"):
    prompt = load_prompt("prompts/yongin.yaml", encoding="utf-8")
    # streaming=True 옵션을 추가하여 스트리밍 응답을 활성화합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0, streaming=True)
    chain = {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return chain


# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chain" not in st.session_state:
    chain = create_chain(model_name="gpt-4o")
    st.session_state["chain"] = chain

# 사이드바: 초기화 버튼과 모델 선택 메뉴
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o-mini", "gpt-4-turbo", "gpt-4o"], index=0
    )

if clear_btn:
    st.session_state["messages"] = []

print_messages()

# 사용자 입력 처리 (챗 입력)
user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning_msg = st.empty()

if user_input:
    chain = st.session_state["chain"]
    if chain is not None:
        st.chat_message("user").write(user_input)
        # 스트리밍 방식으로 챗봇 응답 생성
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            # for 루프를 통해 토큰 단위로 출력
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_msg.error("체인을 생성할 수 없습니다.")
