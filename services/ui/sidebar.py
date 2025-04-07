import streamlit as st

from services.constants import CONTENT_NOT_IN_TEXTBOOK


def set_active_with_page(question_id: str, book_name: str, page_no: int):
    """페이지 번호와 함께 모달 활성화"""
    st.session_state["book_names"][question_id] = book_name
    st.session_state["modal_current_page"][question_id] = int(page_no)
    st.session_state["active_question_id"] = question_id


def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.header("대화 기록")

        # 대화 기록이 있는 경우에만 표시
        if len(st.session_state["messages"]) > 1:  # 첫 메시지는 시스템 메시지
            messages = st.session_state["messages"]

            for i, msg in enumerate(messages):
                if msg.role == "user":
                    q_id = None
                    for id, q in st.session_state["questions"].items():
                        if q == msg.content:
                            q_id = id
                            break

                    if q_id:
                        with st.expander(f"Q: {msg.content}", expanded=True):
                            # 검색 결과 가져오기
                            results = st.session_state["question_results"].get(q_id, [])

                            # 응답 찾기
                            question_idx = i
                            if question_idx != -1 and question_idx + 1 < len(messages):
                                # 교재에서 답을 찾을 수 없는 경우
                                if not st.session_state.get("can_answer", {}).get(
                                    q_id, False
                                ):
                                    st.write(CONTENT_NOT_IN_TEXTBOOK)
                                # 교재에서 답을 찾은 경우
                                elif results:
                                    st.write("📝 참고 페이지")
                                    for idx, result in enumerate(results[:3]):
                                        page_no = result.get("page_no")
                                        book_name = result.get("metadata", {}).get(
                                            "title"
                                        )

                                        st.button(
                                            f"📖 {book_name} {page_no}p",
                                            key=f"page_btn_{q_id}_{idx}",
                                            on_click=lambda q_id=q_id, b_name=book_name, p_no=page_no: set_active_with_page(
                                                q_id, b_name, p_no
                                            ),
                                            use_container_width=True,
                                        )
