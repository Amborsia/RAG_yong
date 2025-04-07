import json
import os
import re
from pathlib import Path


def clean_page_content(content: str) -> str:
    """
    페이지 내용에서 메타데이터를 제거하고 정리합니다.

    Args:
        content: 페이지 내용
    Returns:
        정리된 페이지 내용
    """
    # 줄 단위로 분리
    lines = content.split("\n")

    # 메타데이터 라인 제거
    cleaned_lines = [
        line
        for line in lines
        if not line.strip().startswith("18EBS_뉴런_중학과학미니북")
    ]

    # 정리된 내용 반환
    return "\n".join(cleaned_lines).strip()


def convert_text_to_json(text_file_path: str, output_file_path: str) -> None:
    """
    ..PAGE: 구분자로 구분된 텍스트 파일을 JSON 형식으로 변환합니다.

    Args:
        text_file_path: 입력 텍스트 파일 경로
        output_file_path: 출력 JSON 파일 경로
    """
    print(f"Converting {text_file_path} to {output_file_path}")

    with open(text_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # ..PAGE: 구분자로 텍스트를 분할
    pages = content.split("..PAGE:")

    # 첫 번째 요소는 빈 문자열이므로 제거
    if pages[0].strip() == "":
        pages = pages[1:]

    # 파일 이름에서 확장자를 제외한 부분을 타이틀로 사용
    title = Path(text_file_path).stem
    book_data = {"title": title, "pages": {}}

    for page in pages:
        if not page.strip():
            continue

        # 페이지 번호와 내용 분리
        page_parts = page.strip().split("\n", 1)
        if len(page_parts) == 2:
            page_no = page_parts[0].strip()
            content = clean_page_content(page_parts[1].strip())
            book_data["pages"][page_no] = content

    # JSON 파일로 저장
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(book_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully converted {text_file_path}")


def process_directory(directory_path: str) -> None:
    """
    지정된 디렉토리의 모든 .txt 파일을 .json 파일로 변환합니다.

    Args:
        directory_path: 처리할 디렉토리 경로
    """
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Directory not found: {directory_path}")
        return

    # 모든 .txt 파일 찾기
    txt_files = list(directory.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {directory_path}")
        return

    print(f"Found {len(txt_files)} text files to process")

    # 각 텍스트 파일을 JSON으로 변환
    for txt_file in txt_files:
        json_file = txt_file.with_suffix(".json")
        convert_text_to_json(str(txt_file), str(json_file))


def main():
    directory_path = "data/ebs/texts"
    process_directory(directory_path)


if __name__ == "__main__":
    main()
