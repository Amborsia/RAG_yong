import json
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()


class EbsRAG:
    def __init__(self):
        self.vs2_url = os.getenv("VS2_URL")
        self.vs2_model = os.getenv("VS2_MODEL")
        self.index_name = os.getenv("VS2_COLLECTION_NAME")

        # JSON 파일들이 있는 디렉토리 경로
        self.texts_dir = "data/ebs/chunks"
        # JSON 데이터를 메모리에 캐시
        self.book_data_cache = {}

    def _load_book_data(self, book_name: str) -> Dict[str, Any]:
        """
        책 제목에 해당하는 JSON 데이터를 로드합니다.
        캐시된 데이터가 있으면 캐시에서 반환하고, 없으면 파일에서 로드합니다.
        """
        if book_name not in self.book_data_cache:
            json_path = os.path.join(self.texts_dir, f"{book_name}.json")
            try:
                print(f"\n[DEBUG] Loading book: {book_name}")
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # 리스트 형태의 데이터를 pages 딕셔너리로 변환
                    pages = {}
                    for item in data:
                        page_no = item.get("pageNo")
                        if page_no:
                            # contents 배열의 첫 번째 항목을 페이지 내용으로 사용
                            contents = item.get("contents", [])
                            if contents:
                                pages[page_no] = contents[0]

                    self.book_data_cache[book_name] = {"pages": pages}

            except FileNotFoundError:
                print(f"[WARNING] JSON file not found: {book_name}")
                return {"pages": {}}
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON decode error in {book_name}.json: {e}")
                return {"pages": {}}
        return self.book_data_cache[book_name]

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        VS2 API를 사용하여 질문과 관련된 페이지를 검색합니다.

        Args:
            query: 사용자 질문
            top_k: 반환할 최대 결과 수

        Returns:
            검색된 페이지
        """
        try:
            url = f"{self.vs2_url}/{self.vs2_model}/{self.index_name}/_search"
            payload = {"sentences": [query]}
            headers = {"Content-Type": "application/json"}

            print(f"\n[DEBUG] Searching for: {query}")
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            results = response.json()
            enriched_results = []

            for hit in results.get("_objects", [])[:top_k]:
                try:
                    metadata = hit["_metadata"]
                    page_no = metadata.get("pageNo")
                    if not page_no:
                        print(f"[WARNING] No page number in metadata")
                        continue

                    book_name = metadata["title"]
                    print(f"[DEBUG] Found in {book_name} (page {page_no})")

                    # 해당 책의 JSON 데이터 로드
                    book_data = self._load_book_data(book_name)
                    page_content = book_data["pages"].get(page_no, "")

                    if not page_content:
                        print(
                            f"[WARNING] No content found for page {page_no} in {book_name}"
                        )
                        continue

                    enriched_results.append(
                        {
                            "score": hit["_score"],
                            "page_no": page_no,
                            "content": page_content,
                            "metadata": metadata,
                            "book_name": book_name,
                        }
                    )
                except KeyError as e:
                    print(f"[ERROR] Key error: {e}")
                    continue

            return enriched_results

        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []

    def search_vs2(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        VS2 API를 사용하여 질문과 관련된 페이지를 검색합니다.

        Args:
            query: 사용자 질문
            top_k: 반환할 최대 결과 수

        Returns:
            검색된 페이지
        """
        # Implementation of the search_vs2 method
        pass

        return []  # Placeholder return, actual implementation needed

    def search_vs2_with_context(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        컨텍스트를 포함한 향상된 검색을 수행합니다.
        """
        results = self.search(query, top_k)

        # 연속된 페이지 확인 및 컨텍스트 확장
        enriched_results = []
        for result in results:
            page_no = result["page_no"]  # 문자열로 유지
            book_name = result["book_name"]

            # 해당 책의 JSON 데이터 로드
            book_data = self._load_book_data(book_name)

            # 이전/다음 페이지 컨텍스트 추가
            # 현재 페이지 번호를 정수로 변환하여 이전/다음 페이지 계산
            current_page = int(page_no)
            adjacent_pages = {
                str(current_page - 1): book_data["pages"].get(str(current_page - 1)),
                str(current_page + 1): book_data["pages"].get(str(current_page + 1)),
            }

            result["adjacent_context"] = adjacent_pages
            enriched_results.append(result)

        return enriched_results

    def search_vs2_with_context_and_summary(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        VS2 API를 사용하여 질문과 관련된 페이지를 검색합니다.

        Args:
            query: 사용자 질문
            top_k: 반환할 최대 결과 수

        Returns:
            검색된 페이지
        """
        # Implementation of the search_vs2_with_context_and_summary method
        pass

        return []  # Placeholder return, actual implementation needed

    def search_vs2_with_context_and_summary_and_answer(
        self, query: str, top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        VS2 API를 사용하여 질문과 관련된 페이지를 검색합니다.

        Args:
            query: 사용자 질문
            top_k: 반환할 최대 결과 수

        Returns:
            검색된 페이지
        """
        # Implementation of the search_vs2_with_context_and_summary_and_answer method
        pass

        return []  # Placeholder return, actual implementation needed
