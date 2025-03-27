import logging
import os
import uuid  # UUID 생성 모듈 추가

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI

from services.ebs import EbsRAG
from services.pdf_viewer import PDFViewer
from utils.greeting_message import GREETING_MESSAGE
from utils.prompts import load_prompt

ebs_rag = EbsRAG()

pdf = PDFViewer("data/ebs/pdfs/뉴런과학1_미니북.pdf")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 스타일 설정
st.markdown(
    """
<style>
.stMain { position: relative; }
.stChatMessage { background-color: transparent !important; }
.st-emotion-cache-glsyku { align-items: center; }
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
</style>
""",
    unsafe_allow_html=True,
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content=GREETING_MESSAGE["ebs_tutor"])
    ]
if "pdf_viewer_state" not in st.session_state:
    st.session_state["pdf_viewer_state"] = {"current_page": None}
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None
if "questions" not in st.session_state:
    st.session_state["questions"] = {}  # 질문 목록 초기화
if "sources" not in st.session_state:
    st.session_state["sources"] = {}  # 참고 페이지 정보를 저장할 키
if "modal_open" not in st.session_state:
    st.session_state["modal_open"] = False  # 모달 열림 상태

st.title("EBS 과학 튜터 챗봇")

# 이전 대화 메시지(인사말 포함) 표시
for message in st.session_state["messages"]:
    st.chat_message(message.role).write(message.content)


# 검색 결과 필터링 함수 추가
def filter_results(results):
    return [
        r
        for r in results
        if r.get("score", 0) >= 0.5 and len(r.get("content", "").strip()) > 30
    ]


if user_input := st.chat_input("궁금한 내용을 물어보세요!", key="chat_input"):
    question_id = str(uuid.uuid4())  # 고유한 질문 ID 생성
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    st.session_state["questions"][question_id] = user_input

    try:
        results = ebs_rag.search(user_input, top_k=3)
        st.session_state["search_results"] = results

        if results:
            # 컨텍스트 구성
            context_chunks = []
            sources = []
            for r in results:
                page_no = r.get("page_no")
                content = r.get("content")
                if page_no and content:
                    context_chunks.append(f"[{page_no}페이지]\n{content}")
                    sources.append(f"{page_no}페이지")
            context_text = "\n\n".join(context_chunks)
            if context_chunks:
                st.session_state["pdf_viewer_state"]["current_page"] = results[0][
                    "page_no"
                ]
        else:
            context_text = "관련 내용을 찾지 못했습니다."
            sources = []
    except Exception as e:
        st.error(f"검색 중 오류 발생: {str(e)}")
        context_text = "검색 중 오류가 발생했습니다."
        sources = []

    prompt_template = load_prompt("prompts/ebs_tutor.yaml")
    formatted_prompt = prompt_template.format(
        context_text=context_text, question=user_input
    )

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    response_generator = llm.stream(formatted_prompt)
    response_text = ""

    with st.chat_message("assistant"):
        response_container = st.empty()
        for chunk in response_generator:
            chunk_text = getattr(chunk, "content", str(chunk))
            response_text += chunk_text
            response_container.markdown(response_text)

    st.session_state["messages"].append(
        ChatMessage(role="assistant", content=response_text)
    )
    st.session_state["sources"][question_id] = sources
    st.session_state["pdf_viewer_state"]["current_page"] = (
        results[0]["page_no"] if results else None
    )


@st.dialog("참고 페이지 내용")
def pdf_viewer_modal(item):
    pdf_path = "data/ebs/pdfs/뉴런과학1_미니북.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
    else:
        try:
            current_page = item
            pdf.show_pdf(int(current_page))
        except Exception as e:
            st.error(f"PDF 표시 중 오류 발생: {str(e)}")


# 사이드바 표시
with st.sidebar:
    st.write("## 📌 질문 목록 및 참고 페이지")
    if "sources" in st.session_state and "questions" in st.session_state:
        for q_id, source_list in st.session_state["sources"].items():
            if q_id in st.session_state["questions"]:
                question_text = st.session_state["questions"][q_id]
                with st.expander(
                    f"💬 {len(question_text) > 30 and question_text[:30] + '...' or question_text}"
                ):
                    st.write(
                        f"📝 참고 페이지:\n{', '.join([source.replace('페이지', 'p') for source in source_list])}"
                    )
                    if st.button("📖 교재 보기", key=f"show_reference_page_{q_id}"):
                        # 모달 열림 상태를 관리
                        if st.session_state.get("modal_open", False):
                            st.warning("이미 모달이 열려 있습니다. 먼저 닫아주세요.")
                        else:
                            st.session_state["modal_open"] = True
                            current_page = st.session_state["pdf_viewer_state"].get(
                                "current_page", None
                            )
                            if current_page is not None:
                                pdf_viewer_modal(current_page)
                            # 모달이 닫히면(함수 실행이 끝나면) 상태 초기화
                            st.session_state["modal_open"] = False
