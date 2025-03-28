import os

import streamlit as st


class PDFViewer:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = self.load_image_files()
        self.total_pages = len(self.image_files)

        # 뷰어별 고유 세션 상태 키 생성
        self.session_key = f"pdf_page_{hash(self.image_dir)}"

        # 세션 상태 초기화
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = 1

    def load_image_files(self):
        # PNG 파일만 목록에 포함시키고, 파일명을 자연스러운 순서로 정렬합니다.
        if not os.path.exists(self.image_dir):
            return []

        files = [f for f in os.listdir(self.image_dir) if f.endswith(".png")]
        # 파일명에서 숫자 부분을 추출해 정렬 (예: "page_1.png", "page_2.png", ...)
        files.sort(key=lambda x: int("".join(filter(str.isdigit, x)) or 0))
        return files

    def render_viewer(self):
        # 총 페이지 수가 0이면 안내 메시지 출력
        if self.total_pages == 0:
            st.error("불러올 PDF 페이지 이미지가 없습니다.")
            return

        # 현재 페이지 가져오기
        current_page = st.session_state[self.session_key]

        # 현재 페이지에 해당하는 이미지 표시
        try:
            image_path = os.path.join(
                self.image_dir, self.image_files[current_page - 1]
            )
            st.image(image_path, use_column_width=True)
        except (IndexError, FileNotFoundError):
            st.error(f"페이지 {current_page}를 불러올 수 없습니다.")

        # 이전 및 다음 페이지로 이동하는 함수 정의
        def go_prev():
            st.session_state[self.session_key] = max(1, current_page - 1)

        def go_next():
            st.session_state[self.session_key] = min(self.total_pages, current_page + 1)

        # 이전/다음 페이지 버튼을 상단에 배치
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            prev_disabled = current_page <= 1
            if st.button(
                "◀️ 이전", disabled=prev_disabled, key=f"prev_{self.session_key}"
            ):
                go_prev()

        with col2:
            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 8px;">
                    <span style="font-weight: bold;">페이지 {current_page} / {self.total_pages}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            next_disabled = current_page >= self.total_pages
            if st.button(
                "다음 ▶️", disabled=next_disabled, key=f"next_{self.session_key}"
            ):
                go_next()
