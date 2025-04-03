import streamlit as st

from services.constants import NOT_FOUND_IN_TEXTBOOK


def set_active_with_page(question_id: str, book_name: str, page_no: int):
    """페이지 번호와 함께 모달 활성화"""
    st.session_state["book_names"][question_id] = book_name
    st.session_state["modal_current_page"][question_id] = int(page_no)
    st.session_state["active_question_id"] = question_id


def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.write("## 📌 질문 목록 및 참고 페이지")
        for q_id, source_list in st.session_state["sources"].items():
            if q_id in st.session_state["questions"]:
                question_text = st.session_state["questions"][q_id]
                display_text = (
                    f"{question_text[:30]}..."
                    if len(question_text) > 30
                    else question_text
                )

                with st.expander(f"💬 {display_text}"):
                    results = st.session_state["question_results"].get(q_id, [])
                    messages = st.session_state.get("messages", [])

                    # 해당 질문에 대한 응답 찾기
                    question_idx = next(
                        (
                            i
                            for i, msg in enumerate(messages)
                            if msg.role == "user"
                            and msg.content == st.session_state["questions"][q_id]
                        ),
                        -1,
                    )
                    if question_idx != -1 and question_idx + 1 < len(messages):
                        response = messages[question_idx + 1].content
                        if "찾을 수 없는 내용이에요" in response:
                            st.write(NOT_FOUND_IN_TEXTBOOK)
                        elif results:
                            st.write("📝 참고 페이지")
                            for idx, result in enumerate(results[:3]):
                                page_no = result.get("page_no")
                                book_name = result.get("metadata", {}).get("title")

                                st.button(
                                    f"📖 {book_name} {page_no}p",
                                    key=f"page_btn_{q_id}_{idx}",
                                    on_click=lambda q_id=q_id, b_name=book_name, p_no=page_no: set_active_with_page(
                                        q_id, b_name, p_no
                                    ),
                                    use_container_width=True,
                                )
