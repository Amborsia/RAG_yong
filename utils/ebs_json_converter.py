import json
import re


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
    with open(text_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # ..PAGE: 구분자로 텍스트를 분할
    pages = content.split("..PAGE:")

    # 첫 번째 요소는 빈 문자열이므로 제거
    if pages[0].strip() == "":
        pages = pages[1:]

    book_data = {"title": "뉴런과학1_미니북", "pages": {}}

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


def main():
    text_file_path = "data/ebs/texts/뉴런과학1_미니북.txt"
    output_file_path = "data/ebs/texts/뉴런과학1_미니북.json"
    convert_text_to_json(text_file_path, output_file_path)


if __name__ == "__main__":
    main()
