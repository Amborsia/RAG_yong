import logging
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI

from services.ebs import EbsRAG
from services.pdf_viewer import PDFViewer
from utils.greeting_message import GREETING_MESSAGE
from utils.prompts import load_prompt

# EBS 검색 객체 생성
ebs_rag = EbsRAG()

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
    st.session_state["sources"] = {}  # 참고 페이지 정보를 저장할 딕셔너리
if "modal_open" not in st.session_state:
    st.session_state["modal_open"] = False  # 모달 열림 상태
if "pdf_viewer" not in st.session_state:
    st.session_state["pdf_viewer"] = None  # PDF 뷰어 인스턴스 저장용
if "modal_current_page" not in st.session_state:
    st.session_state["modal_current_page"] = {}
if "active_question_id" not in st.session_state:
    st.session_state["active_question_id"] = None
if "pdf_viewer_directories" not in st.session_state:
    st.session_state["pdf_viewer_directories"] = {}  # 책 이름별 이미지 디렉토리 저장
if "book_names" not in st.session_state:
    st.session_state["book_names"] = {}  # 질문 ID별 책 이름 저장
if "question_results" not in st.session_state:
    st.session_state["question_results"] = {}  # 질문별 검색 결과 저장

# 앱 메인 타이틀 및 기존 대화 표시
st.title("EBS 과학 튜터 챗봇")

for message in st.session_state["messages"]:
    st.chat_message(message.role).write(message.content)


# 검색 결과 필터링 함수 (예시)
def filter_results(results):
    filtered_results = []
    for r in results:
        if r.get("score", 0) >= 0.5 and len(r.get("content", "").strip()) > 30:
            book_name = r.get("book_name") or r.get("metadata", {}).get("title")
            filtered_results.append((r, book_name))
    return filtered_results


# 사용자 입력 처리
if user_input := st.chat_input("궁금한 내용을 물어보세요!", key="chat_input"):
    question_id = str(uuid.uuid4())
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    st.session_state["questions"][question_id] = user_input

    try:
        results = ebs_rag.search(user_input, top_k=3)
        st.session_state["search_results"] = results
        # 질문별로 검색 결과 저장
        st.session_state["question_results"][question_id] = results

        if results:
            # 컨텍스트 구성
            context_chunks = []
            sources = []
            for r, book_name in filter_results(results):
                page_no = r.get("page_no")
                content = r.get("content")
                if page_no and content:
                    context_chunks.append(f"[{page_no}페이지]\n{content}")
                    sources.append(f"{page_no}페이지")
                st.session_state["book_names"][question_id] = book_name  # 책 이름 저장
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


# 각 질문별 모달 전용 페이지 번호를 별도로 관리
if "modal_current_page" not in st.session_state:
    st.session_state["modal_current_page"] = {}


# 사이드바에서 버튼 클릭 시 각 질문의 시작 페이지 번호를 저장
def set_active(question_id, book_name):
    # 책 이름 저장
    st.session_state["book_names"][question_id] = book_name
    if question_id not in st.session_state["modal_current_page"]:
        st.session_state["modal_current_page"][question_id] = int(
            st.session_state["pdf_viewer_state"].get("current_page", 1)
        )
    st.session_state["active_question_id"] = question_id


# 모달 함수: 전달된 질문 ID(q_id)에 해당하는 페이지 번호를 사용
@st.dialog("참고 자료")
def pdf_viewer_modal(q_id):
    book_name = st.session_state["book_names"].get(q_id)
    if not book_name:
        st.write(f"책 이름을 찾을 수 없습니다.")
        return

    # 모달 제목 변경
    st.markdown(f"### {book_name.replace('_', ' ')}")

    image_dir = f"data/ebs/pages/{book_name}"
    try:
        pdf_viewer = PDFViewer(image_dir, book_name)
    except ValueError as e:
        st.error(f"PDF 뷰어 초기화 실패: {str(e)}")
        return

    image_container = st.empty()
    page_info = st.empty()

    def update_view():
        current_page = st.session_state["modal_current_page"].get(q_id, 1)

        # 페이지 범위 확인
        if current_page < 1:
            current_page = 1
        elif current_page > pdf_viewer.total_pages:
            current_page = pdf_viewer.total_pages

        try:
            current_image_path = os.path.join(
                pdf_viewer.image_dir, pdf_viewer.image_files[current_page - 1]
            )
            image_container.image(current_image_path, width=600)
            # 페이지 정보 업데이트
            page_info.markdown(
                f"<div style='text-align: center'>{current_page}/{pdf_viewer.total_pages}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"이미지 로딩 실패: {str(e)}")

    update_view()

    # 네비게이션 버튼을 함수 밖으로 이동
    col_prev, col_page, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀ 이전", key=f"modal_prev_{q_id}"):
            current_page = st.session_state["modal_current_page"].get(q_id, 1)
            if current_page > 1:
                st.session_state["modal_current_page"][q_id] = current_page - 1
                update_view()
    # 가운데 열은 비워둠 (페이지 정보가 위에서 표시됨)
    with col_next:
        if st.button("다음 ▶", key=f"modal_next_{q_id}"):
            current_page = st.session_state["modal_current_page"].get(q_id, 1)
            if current_page < pdf_viewer.total_pages:
                st.session_state["modal_current_page"][q_id] = current_page + 1
                update_view()


# 페이지 번호를 포함하여 모달을 활성화하는 함수
def set_active_with_page(question_id, book_name, page_no):
    st.session_state["book_names"][question_id] = book_name
    st.session_state["modal_current_page"][question_id] = int(page_no)
    st.session_state["active_question_id"] = question_id


with st.sidebar:
    st.write("## 📌 질문 목록 및 참고 페이지")
    if "sources" in st.session_state and "questions" in st.session_state:
        for q_id, source_list in st.session_state["sources"].items():
            if q_id in st.session_state["questions"]:
                question_text = st.session_state["questions"][q_id]
                display_text = (
                    question_text[:30] + "..."
                    if len(question_text) > 30
                    else question_text
                )
                with st.expander(f"💬 {display_text}"):
                    # 해당 질문의 검색 결과 가져오기
                    results = st.session_state["question_results"].get(q_id, [])
                    if results and len(results) > 0:
                        st.write("📝 참고 페이지")
                        # 각 결과를 한 줄로 표시
                        for idx, result in enumerate(results[:3]):
                            page_no = result.get("page_no")
                            book_name = result.get("metadata", {}).get("title")
                            # score = result.get("score", 0)
                            # similarity = int(score * 100)

                            st.button(
                                f"📖 {book_name} {page_no}p",
                                key=f"page_btn_{q_id}_{idx}",
                                on_click=lambda q_id=q_id, b_name=book_name, p_no=page_no: set_active_with_page(
                                    q_id, b_name, p_no
                                ),
                                use_container_width=True,
                            )

# --- 모달 호출: active_question_id가 설정된 경우에만 모달을 열고, 열렸으면 바로 초기화 ---
if st.session_state.get("active_question_id") is not None:
    # 모달을 열고 나면, 이후 다시 같은 모달이 열리지 않도록 active_question_id를 초기화합니다.
    pdf_viewer_modal(st.session_state["active_question_id"])
    st.session_state["active_question_id"] = None
