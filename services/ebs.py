import json
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

from services.pdf_viewer import PDFViewer

load_dotenv()


class EbsRAG:
    def __init__(self):
        self.vs2_url = os.getenv("VS2_URL")
        self.vs2_model = os.getenv("VS2_MODEL")
        self.index_name = "ebs-mini"

        # PDF 뷰어 매니저 초기화
        # self.pdf_viewer = PDFViewer()

        # JSON 데이터 로드
        with open("data/ebs/texts/뉴런과학1_미니북.json", "r", encoding="utf-8") as f:
            self.book_data = json.load(f)

    def get_page_metadata(self, page_no: str) -> Dict[str, Any]:
        """
        페이지에 대한 메타데이터를 반환합니다.
        """
        return {
            "pdf_path": "cache/pdf_pages/뉴런과학1_미니북",
            "page_no": page_no,
            "title": self.book_data["title"],
            # PDF 뷰어에 필요한 추가 메타데이터
            "viewport": {"width": 595, "height": 842},  # A4 기준
        }

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

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            results = response.json()
            enriched_results = []

            for hit in results.get("_objects", [])[:top_k]:
                try:
                    metadata = hit["_metadata"]
                    page_no = metadata["pageNo"]
                    book_name = metadata["title"]  # metadata에서 title을 가져옴

                    # 해당 책의 페이지 내용 가져오기
                    page_content = self.book_data["pages"].get(page_no, "")

                    enriched_results.append(
                        {
                            "score": hit["_score"],
                            "page_no": page_no,
                            "content": page_content,
                            "metadata": metadata,
                            "book_name": book_name,  # 실제 책 제목 사용
                        }
                    )
                except KeyError as e:
                    print(f"검색 결과 처리 중 오류 발생: {e}")
                    continue

            return enriched_results

        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
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
            page_no = int(result["page_no"])

            # 이전/다음 페이지 컨텍스트 추가
            adjacent_pages = {
                str(page_no - 1): self.book_data["pages"].get(str(page_no - 1)),
                str(page_no + 1): self.book_data["pages"].get(str(page_no + 1)),
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
