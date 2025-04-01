import os

import streamlit as st

from services.pdf_viewer import PDFViewer


def apply_modal_styles():
    """모달 스타일 적용"""
    st.markdown(
        """
        <style>
        /* 모달 너비 조정 */
        [aria-label="dialog"] {
            min-width: 800px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def update_pdf_view(image_container, page_info, pdf_viewer, q_id: str):
    """PDF 뷰어 업데이트"""
    current_page = st.session_state["modal_current_page"].get(q_id, 1)

    # 페이지 범위 검증
    current_page = max(1, min(current_page, pdf_viewer.total_pages))

    try:
        current_image_path = os.path.join(
            pdf_viewer.image_dir, pdf_viewer.image_files[current_page - 1]
        )
        # 이미지 너비도 모달 크기에 맞게 조정
        image_container.image(current_image_path, width=800)
        page_info.markdown(
            f"<div style='text-align: center'>{current_page}/{pdf_viewer.total_pages}</div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"이미지 로딩 실패: {str(e)}")


@st.dialog("참고 자료")
def pdf_viewer_modal(q_id: str):
    """PDF 뷰어 모달"""
    # 모달 스타일 적용
    apply_modal_styles()

    book_name = st.session_state["book_names"].get(q_id)
    if not book_name:
        st.write("책 이름을 찾을 수 없습니다.")
        return

    st.markdown(f"### {book_name.replace('_', ' ')}")

    try:
        pdf_viewer = PDFViewer(f"data/ebs/pages/{book_name}", book_name)
    except ValueError as e:
        st.error(f"PDF 뷰어 초기화 실패: {str(e)}")
        return

    image_container = st.empty()
    page_info = st.empty()

    update_pdf_view(image_container, page_info, pdf_viewer, q_id)

    # 네비게이션 컨트롤
    col_prev, col_page, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀ 이전", key=f"modal_prev_{q_id}"):
            current_page = st.session_state["modal_current_page"].get(q_id, 1)
            if current_page > 1:
                st.session_state["modal_current_page"][q_id] = current_page - 1
                update_pdf_view(image_container, page_info, pdf_viewer, q_id)
    with col_next:
        if st.button("다음 ▶", key=f"modal_next_{q_id}"):
            current_page = st.session_state["modal_current_page"].get(q_id, 1)
            if current_page < pdf_viewer.total_pages:
                st.session_state["modal_current_page"][q_id] = current_page + 1
                update_pdf_view(image_container, page_info, pdf_viewer, q_id)
