from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import requests
from pydantic import ValidationError
from pydantic_settings import BaseSettings
from requests.adapters import HTTPAdapter, Retry

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


class AppSettings(BaseSettings):
    vs2_url: str
    vs2_model: str
    vs2_collection_name: str
    texts_dir: str = "data/ebs/chunks"
    gemma_url: str
    gemma_token: str

    # 추가된 환경 변수 필드
    openai_api_key: str
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str
    langchain_project: str = "Yong-in RAG"
    port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class PageContent(TypedDict):
    content: str
    metadata: dict


class SearchResult(TypedDict):
    score: float
    page_no: str
    content: str
    metadata: dict
    book_name: str
    adjacent_context: Optional[Dict[str, str]]


class EbsRAG:
    def __init__(self):
        self.settings = self._load_settings()
        self.session = self._configure_session()
        self._validate_paths()

    def _load_settings(self) -> AppSettings:
        try:
            return AppSettings()
        except ValidationError as e:
            logger.critical("환경 변수 검증 실패: %s", e)
            raise

    def _configure_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
        )
        session.mount("http://", HTTPAdapter(max_retries=retries))
        return session

    def _validate_paths(self) -> None:
        if not Path(self.settings.texts_dir).exists():
            logger.error("텍스트 디렉토리 존재하지 않음: %s", self.settings.texts_dir)
            raise FileNotFoundError(f"Directory not found: {self.settings.texts_dir}")

    @lru_cache(maxsize=100)
    def _load_book_data(self, book_name: str) -> Dict[str, Any]:
        """LRU 캐시를 사용한 책 데이터 로딩 최적화"""
        json_path = Path(self.settings.texts_dir) / f"{book_name}.json"

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return self._process_book_data(book_name, data)

        except FileNotFoundError:
            logger.warning("JSON 파일을 찾을 수 없음: %s", book_name)
            return {"pages": {}}
        except json.JSONDecodeError as e:
            logger.error("JSON 파싱 오류 (%s): %s", book_name, e)
            return {"pages": {}}

    def _process_book_data(self, book_name: str, data: list) -> Dict[str, Any]:
        """JSON 데이터 처리 파이프라인"""
        pages = {}
        for idx, item in enumerate(data, 1):
            try:
                page_no = str(item.get("pageNo", idx))
                contents = item.get("contents", [])
                pages[page_no] = contents[0] if contents else ""
            except (KeyError, IndexError) as e:
                logger.warning(
                    "데이터 처리 오류 (%s 페이지 %s): %s", book_name, page_no, e
                )
        return {"pages": pages}

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """개선된 검색 메소드: 타입 힌트 강화 및 에러 처리 개선"""
        try:
            url = f"{self.settings.vs2_url}/{self.settings.vs2_model}/{self.settings.vs2_collection_name}/_search"
            response = self.session.post(
                url,
                json={"sentences": [query]},
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()

            return self._process_search_results(response.json(), top_k)

        except requests.RequestException as e:
            logger.error("검색 요청 실패: %s", e)
            return []
        except KeyError as e:
            logger.error("응답 데이터 구조 오류: %s", e)
            return []

    def _process_search_results(self, results: dict, top_k: int) -> List[SearchResult]:
        """검색 결과 처리 파이프라인"""
        enriched_results = []
        for hit in results.get("_objects", [])[:top_k]:
            try:
                metadata = hit["_metadata"]
                if not (page_no := metadata.get("pageNo")):
                    logger.warning("페이지 번호 없음: %s", metadata)
                    continue

                book_name = metadata["title"]
                page_content = self._get_page_content(book_name, page_no)
                if not page_content:
                    continue

                enriched_results.append(
                    SearchResult(
                        score=hit["_score"],
                        page_no=page_no,
                        content=page_content,
                        metadata=metadata,
                        book_name=book_name,
                        adjacent_context=None,
                    )
                )
            except KeyError as e:
                logger.error("필드 누락: %s", e)
        return enriched_results

    def _get_page_content(self, book_name: str, page_no: str) -> Optional[str]:
        """페이지 내용 추출 유틸리티 메소드"""
        book_data = self._load_book_data(book_name)
        if content := book_data["pages"].get(page_no):
            return content
        logger.warning("페이지 내용 없음: %s 페이지 %s", book_name, page_no)
        return None

    def search_with_context(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """컨텍스트 추가 검색 메소드"""
        results = self.search(query, top_k)
        for result in results:
            result["adjacent_context"] = self._get_adjacent_pages(
                result["book_name"], result["page_no"]
            )
        return results

    def _get_adjacent_pages(self, book_name: str, page_no: str) -> Dict[str, str]:
        """인접 페이지 컨텍스트 추출"""
        try:
            current_page = int(page_no)
            book_data = self._load_book_data(book_name)
            return {
                str(current_page - 1): book_data["pages"].get(
                    str(current_page - 1), ""
                ),
                str(current_page + 1): book_data["pages"].get(
                    str(current_page + 1), ""
                ),
            }
        except ValueError:
            logger.warning("유효하지 않은 페이지 번호 형식: %s", page_no)
            return {}
