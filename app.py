# app.py
import os
import pickle
import textwrap
import streamlit as st
import models.database as db

from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI
from initialize import init_rag
from services.search import search_top_k

# 간단한 로깅은 기본 print()로 대체하거나, 필요시 다른 로깅 라이브러리를 사용
print("[Project] Yong-in RAG")

# ★ 수정된 파일 경로 ★
INDEX_FILE = "rag_index/index.faiss"
CHUNKED_FILE = "rag_index/index.pkl"
DATA_DIR = "crawling/output"


GREETING_MESSAGE = textwrap.dedent(
    """\
안녕하세요! 더 나은 삶을 위한 **스마트도시**, 용인시청 챗봇입니다.  

저는 **조직도 정보**를 실시간 안내해 드리고 있어요.  

📌 TIP! 이렇게 질문해 보세요!


  - 민원 담당자 연락처 알려줘
  - 청년 월세지원담당자 연락처 알려줘
  - 청년 취업지원 해주는 담당자 알려줘
"""
)



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

def load_or_create_index():
    if os.path.exists(INDEX_FILE):
        # 문서 데이터 로드
        db.load_data(DATA_DIR)

        # FAISS 인덱스 로드 (새 파일 경로 사용)
        db.load_index(INDEX_FILE, index_type="FLAT")

        # chunked_data 로드 (없으면 경고)
        try:
            with open(CHUNKED_FILE, "rb") as f:
                loaded_chunked = pickle.load(f)
            # loaded_chunked가 tuple이면 첫 번째 요소를 사용합니다.
            if isinstance(loaded_chunked, tuple):
                db.chunked_data = loaded_chunked[0]
            else:
                db.chunked_data = loaded_chunked
        except FileNotFoundError:
            st.warning("⚠️ `chunked_data.pkl` 파일이 없습니다. 일부 기능이 제한될 수 있습니다.")
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
                loaded_chunked = pickle.load(f)
            if isinstance(loaded_chunked, tuple):
                db.chunked_data = loaded_chunked[0]
            else:
                db.chunked_data = loaded_chunked
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


# --- 커스텀 프롬프트 runnable 정의 ---
# 프롬프트 템플릿 파일(prompts/yongin.yml)을 읽어서 사용자 질문을 채워 넣는 역할
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
def create_chain(model_name="gpt-4o-mini"):
    # 프롬프트 파일 경로를 yml 확장자로 변경하여 로드
    prompt_template = load_prompt("prompts/yongin.yaml", encoding="utf-8")
    # 커스텀 프롬프트 runnable 생성
    prompt_runnable = RunnablePrompt(prompt_template)
    # streaming=True 옵션을 추가하여 스트리밍 응답을 활성화합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0, streaming=True)
    # 체인 구성: 초기 입력은 {"question": <사용자 질문>}를 RunnablePassthrough()로 그대로 넘기고,
    # 그 다음 커스텀 프롬프트 runnable로 템플릿 적용, 이후 llm 호출, 마지막에 StrOutputParser()로 결과 파싱.
    chain = {"question": RunnablePassthrough()} | prompt_runnable | llm | StrOutputParser()
    return chain


# 검색 쿼리 재작성
def rewrite_query(user_question: str) -> str:
    """
    LLM을 활용하여 사용자 질문을 바탕으로 용인시청 관련 최신 정보를 포함할 수 있는 검색 쿼리를 동적으로 재작성합니다.
    """
    llm_rewriter = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=False)
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
    chain = create_chain(model_name="gpt-4o-mini")
    st.session_state["chain"] = chain

## 최초 접속 시 챗봇 인사말 자동 추가 (대화가 시작되지 않은 경우)
if not st.session_state["messages"]:
    add_message("assistant", GREETING_MESSAGE)

# 사이드바: 초기화 버튼과 모델 선택 메뉴 (주석 처리된 부분은 필요에 따라 활성화)
selected_model = "gpt-4o-mini"

print_messages()

# 사용자 입력 처리 (챗 입력)
user_input = st.chat_input("궁금한 내용을 물어보세요!")
warning_msg = st.empty()

if user_input:
    chain = st.session_state["chain"]
    if chain is not None:
        st.chat_message("user").write(user_input)

        # 먼저 사용자 입력을 그대로 검색 쿼리로 사용합니다.
        query_for_search = user_input
        results = search_top_k(query_for_search, top_k=5, ranking_mode="rrf")
        print("RESULT: ", results)
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
                enriched_chunk = f"이 chunk는 {doc_url} 에서 가져온 내용입니다.\n{chunk_text}"
                answer_chunks.append(enriched_chunk)
            context_text = "\n\n".join(answer_chunks)
        # --- RAG: end ---

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
        combined_query = (
            f"아래는 관련 문서 내용 (RAG):\n{context_text}\n\n"
            f"{conversation_section}"
            f"최종 질문: {user_input}"
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

        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_msg.error("체인을 생성할 수 없습니다.")