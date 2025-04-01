import os

import streamlit as st


class PDFViewer:
    def __init__(self, image_dir, book_name):
        self.image_dir = image_dir
        self.book_name = book_name

        # 디렉토리 존재 여부 확인
        if not os.path.exists(image_dir):
            raise ValueError(f"Directory not found: {image_dir}")

        # 파일 목록 가져오기
        self.image_files = [f for f in os.listdir(image_dir) if f.startswith("page_")]

        # 페이지 번호로 정렬 (page_1.png, page_2.png, ...)
        self.image_files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))

        self.total_pages = len(self.image_files)

        if self.total_pages == 0:
            raise ValueError(f"No image files found in directory: {image_dir}")

        # print(f"Loaded {self.total_pages} pages from {image_dir}")  # 디버깅용

        # 세션 상태에 현재 페이지 초기화
        if f"{self.book_name}_current_page" not in st.session_state:
            st.session_state[f"{self.book_name}_current_page"] = 1

    def render_viewer(self):
        current_page = st.session_state[f"{self.book_name}_current_page"]
        st.write(f"페이지 {current_page} / {self.total_pages}")

        # 현재 페이지 이미지 파일 경로 설정
        current_image_path = os.path.join(
            self.image_dir, self.image_files[current_page - 1]
        )

        # 이미지 렌더링
        st.image(current_image_path)

        # 이전/다음 버튼 레이아웃
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("이전 페이지") and current_page > 1:
                st.session_state[f"{self.book_name}_current_page"] -= 1
                st.experimental_rerun()
        with col3:
            if st.button("다음 페이지") and current_page < self.total_pages:
                st.session_state[f"{self.book_name}_current_page"] += 1
                st.experimental_rerun()

        # 특정 페이지로 이동하는 기능
        st.write("특정 페이지로 이동:")
        target_page = st.number_input(
            "페이지 번호 입력",
            min_value=1,
            max_value=self.total_pages,
            value=current_page,
            step=1,
        )
        if st.button("이동"):
            st.session_state[f"{self.book_name}_current_page"] = int(target_page)
            st.experimental_rerun()

    def render_modal(self, initial_page=1):
        # 모달에 전달할 초기 페이지 번호를 설정
        if f"{self.book_name}_current_page" not in st.session_state:
            st.session_state[f"{self.book_name}_current_page"] = initial_page

        current_page = st.session_state[f"{self.book_name}_current_page"]
        st.write(f"모달 - 페이지 {current_page} / {self.total_pages}")

        current_image_path = os.path.join(
            self.image_dir, self.image_files[current_page - 1]
        )

        # 모달 내에서 이미지 렌더링
        st.image(current_image_path)

        # 모달 내에서도 이전/다음 버튼을 추가할 수 있음
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if (
                st.button("이전 페이지", key=f"{self.book_name}_modal_prev")
                and current_page > 1
            ):
                st.session_state[f"{self.book_name}_current_page"] -= 1
                st.experimental_rerun()
        with col3:
            if (
                st.button("다음 페이지", key=f"{self.book_name}_modal_next")
                and current_page < self.total_pages
            ):
                st.session_state[f"{self.book_name}_current_page"] += 1
                st.experimental_rerun()
