import argparse
import os
from pathlib import Path

import fitz  # PyMuPDF


def preprocess_pdf(pdf_path: str, output_dir: str = None, dpi: int = 300):
    """
    PDF 파일을 미리 페이지별로 분할하여 이미지로 저장합니다.

    Args:
        pdf_path: PDF 파일 경로
        output_dir: 이미지 저장 경로 (기본값: data/ebs/pages/파일이름)
        dpi: 이미지 해상도 (기본값: 300)

    Returns:
        tuple: (저장된 페이지 수, 저장 경로)
    """
    # PDF 파일 이름 추출 (확장자 제외)
    pdf_name = Path(pdf_path).stem

    # 저장 경로 설정
    if output_dir is None:
        output_dir = f"data/ebs/pages/{pdf_name}"

    # 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # PDF 파일 열기
    doc = fitz.open(pdf_path)
    page_count = len(doc)

    print(f"PDF '{pdf_name}' 분할 중... (총 {page_count}페이지)")

    # 페이지별로 이미지 저장
    for page_num in range(page_count):
        page = doc[page_num]
        # 해상도 설정 (dpi 값이 클수록 고화질)
        zoom = dpi / 72  # 기본 PDF DPI는 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)

        # 이미지 파일 저장
        img_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(img_path)

        # 진행상황 출력
        if (page_num + 1) % 10 == 0 or page_num + 1 == page_count:
            print(f"  - {page_num + 1}/{page_count} 페이지 처리 완료")

    print(f"PDF 분할 완료: {page_count}페이지가 '{output_dir}' 경로에 저장되었습니다.")
    return page_count, output_dir


if __name__ == "__main__":
    # 명령줄에서 실행할 수 있도록 인자 처리
    parser = argparse.ArgumentParser(description="PDF 파일을 페이지별 이미지로 분할")
    parser.add_argument("pdf_path", help="PDF 파일 경로")
    parser.add_argument("--output", help="이미지 저장 경로")
    parser.add_argument(
        "--dpi", type=int, default=300, help="이미지 해상도 (기본값: 300)"
    )

    args = parser.parse_args()
    preprocess_pdf(args.pdf_path, args.output, args.dpi)
