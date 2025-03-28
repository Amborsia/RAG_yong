import os

import streamlit as st


class PDFViewer:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        # 파일 이름에서 페이지 번호를 추출해 정렬 (예: "page_1.png", "page_2.png", ...)
        self.image_files = sorted(
            os.listdir(image_dir),
            key=lambda f: int(os.path.splitext(f)[0].split("_")[1]),
        )
        self.total_pages = len(self.image_files)

    def render_viewer(self):
        # 세션 상태에 현재 페이지 초기화
        if "current_page" not in st.session_state:
            st.session_state.current_page = 1

        st.write(f"페이지 {st.session_state.current_page} / {self.total_pages}")

        # 현재 페이지 이미지 파일 경로 설정
        current_image_path = os.path.join(
            self.image_dir, self.image_files[st.session_state.current_page - 1]
        )

        # 이미지 렌더링
        st.image(current_image_path)

        # 이전/다음 버튼 레이아웃
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("이전 페이지") and st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.experimental_rerun()
        with col3:
            if (
                st.button("다음 페이지")
                and st.session_state.current_page < self.total_pages
            ):
                st.session_state.current_page += 1
                st.experimental_rerun()

        # 특정 페이지로 이동하는 기능
        st.write("특정 페이지로 이동:")
        target_page = st.number_input(
            "페이지 번호 입력",
            min_value=1,
            max_value=self.total_pages,
            value=st.session_state.current_page,
            step=1,
        )
        if st.button("이동"):
            st.session_state.current_page = int(target_page)
            st.experimental_rerun()

    def render_modal(self, initial_page=1):
        # 모달에 전달할 초기 페이지 번호를 설정
        st.session_state.current_page = initial_page

        st.write(f"모달 - 페이지 {st.session_state.current_page} / {self.total_pages}")

        current_image_path = os.path.join(
            self.image_dir, self.image_files[st.session_state.current_page - 1]
        )

        # 모달 내에서 이미지 렌더링
        st.image(current_image_path)

        # 모달 내에서도 이전/다음 버튼을 추가할 수 있음
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if (
                st.button("이전 페이지", key="modal_prev")
                and st.session_state.current_page > 1
            ):
                st.session_state.current_page -= 1
                st.experimental_rerun()
        with col3:
            if (
                st.button("다음 페이지", key="modal_next")
                and st.session_state.current_page < self.total_pages
            ):
                st.session_state.current_page += 1
                st.experimental_rerun()
