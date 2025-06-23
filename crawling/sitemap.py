import json
import logging
import os
import ssl
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter

# --- 설정 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_URL = "https://www.yongin.go.kr"
SITEMAP_URL = "https://www.yongin.go.kr/user/web/sub/sitemap.do"
ALLOWED_DOMAINS = ["www.yongin.go.kr", "lib.yongin.go.kr", "museum.yongin.go.kr"]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sitemap_crawled_documents.json")


# --- [리팩터링] 커스텀 HTTP 어댑터 (기능 동일) ---
class CustomCipherAdapter(HTTPAdapter):
    """TLS 1.2 및 완화된 암호 그룹을 사용하도록 강제하는 커스텀 HTTP 어댑터"""

    def init_poolmanager(self, connections, maxsize, block=False):
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        try:
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        except AttributeError:
            ctx.options |= ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1

        ctx.set_ciphers("DEFAULT:@SECLEVEL=1")
        self.poolmanager = requests.packages.urllib3.poolmanager.PoolManager(
            num_pools=connections, maxsize=maxsize, block=block, ssl_context=ctx
        )


# --- [리팩터링] 헬퍼 함수들 ---


def _parse_title(raw_title: str) -> str:
    """브레드크럼 형식의 제목 문자열을 올바른 순서로 파싱합니다."""
    if " < " not in raw_title:
        return raw_title
    try:
        main_part = raw_title.split(":")[0]
        parts = [part.strip() for part in main_part.split("<")]
        parts.reverse()
        return " > ".join(parts)
    except Exception:
        logging.warning(f"제목 파싱 실패: '{raw_title}'. 원본 제목을 사용합니다.")
        return raw_title


def _clean_content_html(content_div: BeautifulSoup) -> str:
    """본문 HTML에서 불필요한 요소와 속성을 제거하여 정제합니다."""
    if not content_div:
        return ""

    # 1. 경로(breadcrumb) 정보 및 불필요한 태그 제거
    selectors_to_remove = [
        "div.location",
        "p.location",
        "div.path",
        "div.navi",
        "div.m_location",
        "script",
        "style",
        "nav",
        "aside",
        "header",
        "footer",
        "form",
        "button",
        "input",
    ]
    for selector in selectors_to_remove:
        for tag in content_div.select(selector):
            tag.decompose()

    # 2. HTML 주석 제거
    for comment in content_div(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # 3. 불필요한 속성 제거
    allowed_attrs = ["href", "src", "alt"]
    for tag in [content_div] + content_div.find_all(True):
        attrs = list(tag.attrs.keys())
        for attr in attrs:
            if attr not in allowed_attrs:
                del tag[attr]

    # 4. 텍스트 노드 내 불필요한 공백 정리
    for text_node in content_div(text=lambda text: not isinstance(text, Comment)):
        if text_node.parent.name in ["pre", "code"]:
            continue
        cleaned_text = " ".join(text_node.split())
        if cleaned_text:
            text_node.replace_with(cleaned_text)
        else:
            text_node.extract()

    # 5. 비어있는 태그 반복 제거
    for _ in range(5):
        for tag in content_div.find_all(True):
            if tag.name in ["br", "hr", "img"]:
                continue
            if not tag.find_all(True, recursive=False) and not tag.get_text(strip=True):
                tag.decompose()

    # 정제된 HTML을 문자열로 반환
    html_content = str(content_div)
    return os.linesep.join([s for s in html_content.splitlines() if s.strip()])


def _load_existing_data(file_path: str) -> tuple[list, set]:
    """기존에 크롤링한 데이터가 있으면 불러옵니다."""
    if not os.path.exists(file_path):
        return [], set()

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content:
                return [], set()

            all_documents = json.loads(content)
            crawled_urls = {
                doc["metadata"]["source"]
                for doc in all_documents
                if "metadata" in doc and "source" in doc["metadata"]
            }
            logging.info(
                f"기존 파일에서 {len(all_documents)}개의 문서를 불러왔습니다. ({len(crawled_urls)}개 URL)"
            )
            return all_documents, crawled_urls
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        logging.warning(f"{file_path} 처리 중 오류({e}). 새로 시작합니다.")
        return [], set()


# --- 핵심 로직 함수 ---


def get_all_links_from_sitemap(
    session: requests.Session, sitemap_url: str
) -> list[str]:
    """사이트맵 페이지에서 모든 유효한 링크를 추출합니다."""
    try:
        response = session.get(sitemap_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        links = {
            urljoin(BASE_URL, a["href"].strip())
            for a in soup.find_all("a", href=True)
            if a["href"] and not a["href"].strip().startswith(("#", "javascript:"))
        }
        logging.info(f"사이트맵에서 {len(links)}개의 링크를 발견했습니다.")
        return list(links)
    except requests.exceptions.RequestException as e:
        logging.error(f"사이트맵을 가져오는 중 오류 발생: {e}")
        return []


def scrape_page(url: str, session: requests.Session) -> dict | None:
    """단일 페이지를 스크랩하여 내용과 메타데이터를 반환합니다."""
    try:
        if urlparse(url).hostname not in ALLOWED_DOMAINS:
            logging.info(f"외부 도메인 스킵: {url}")
            return None

        logging.info(f"크롤링: {url}")
        response = session.get(url, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        soup = BeautifulSoup(response.content, "html.parser")

        # 제목 파싱
        title_tag = soup.find("title")
        raw_title = title_tag.get_text(strip=True) if title_tag else "제목 없음"
        title = _parse_title(raw_title)

        # 본문 추출 및 정제
        content_selectors = ["div.cts", "div.content", "div#contents", "main"]
        content_div = next(
            (soup.select_one(s) for s in content_selectors if soup.select_one(s)), None
        )

        content_text = _clean_content_html(content_div)

        if not content_text:
            logging.warning(f"콘텐츠를 찾을 수 없음: {url}")
            return None

        return {
            "page_content": content_text,
            "metadata": {"source": url, "title": title},
        }
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP 오류 {e.response.status_code}: {url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Request 오류: {url} - {e}")
    except Exception as e:
        logging.error(f"페이지 처리 중 알 수 없는 오류: {url} - {e}", exc_info=True)
    return None


def main():
    """메인 실행 함수"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_documents, crawled_urls = _load_existing_data(OUTPUT_FILE)

    session = requests.Session()
    session.mount("https://", CustomCipherAdapter())
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    )

    initial_links = get_all_links_from_sitemap(session, SITEMAP_URL)
    links_to_crawl = [link for link in initial_links if link not in crawled_urls]

    if not links_to_crawl and crawled_urls:
        logging.info("모든 링크를 이미 크롤링했습니다. 작업을 종료합니다.")
        return

    logging.info(
        f"총 {len(initial_links)}개 링크 중 {len(crawled_urls)}개는 이미 처리됨. {len(links_to_crawl)}개를 새로 수집합니다."
    )

    # --- [테스트] 5건만 진행 ---
    # 전체 크롤링 시 아래 두 줄을 주석 처리하세요.
    # logging.warning("***** 테스트 모드: 5개 페이지만 크롤링합니다. *****")
    # links_to_crawl = links_to_crawl[:5]

    for i, link in enumerate(links_to_crawl):
        time.sleep(0.5)
        document = scrape_page(link, session)
        if document:
            all_documents.append(document)
            crawled_urls.add(link)

        if (i + 1) % 50 == 0:
            logging.info(
                f"--- {i + 1}/{len(links_to_crawl)} 페이지 처리 완료. 현재까지 총 {len(all_documents)}개 문서 수집 ---"
            )
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(all_documents, f, ensure_ascii=False, indent=4)
            logging.info("--- 중간 저장 완료 ---")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=4)

    logging.info("\n-------------------------------------------")
    logging.info(f"크롤링 완료! 총 {len(all_documents)}개의 문서를 수집했습니다.")
    logging.info(f"결과 파일: {OUTPUT_FILE}")
    logging.info("-------------------------------------------")


if __name__ == "__main__":
    main()
