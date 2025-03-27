import base64
import os

import streamlit as st


@st.cache_data(show_spinner=False)
def load_pdf_data(pdf_path: str) -> str:
    """
    PDF 파일을 읽어 Base64 문자열로 변환합니다.
    이 결과는 캐시되므로, 같은 경로의 파일을 다시 읽을 때는 빠르게 반환됩니다.
    """
    with open(pdf_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class PDFViewer:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.base64_pdf = None  # 캐시된 PDF 데이터

    def load_pdf(self):
        if not os.path.exists(self.pdf_path):
            st.error(f"PDF 파일을 찾을 수 없습니다: {self.pdf_path}")
            return
        try:
            # 캐시된 load_pdf_data() 함수를 사용하여 PDF 데이터를 불러옴
            self.base64_pdf = load_pdf_data(self.pdf_path)
        except Exception as e:
            st.error(f"PDF 로딩 중 오류 발생: {str(e)}")

    def show_pdf(self, page_no: int = 1):
        # 캐시된 PDF 데이터가 없으면 로드
        if self.base64_pdf is None:
            self.load_pdf()
        if self.base64_pdf is None:
            return  # 로딩 실패 시 종료
        pdf_display = f"""
            <iframe
                src="data:application/pdf;base64,{self.base64_pdf}#page={page_no}"
                width="100%"
                height="800px"
                style="border: none;">
            </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
