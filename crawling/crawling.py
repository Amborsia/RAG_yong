import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import re
import os

visited = set()

# def convert_html_to_text_with_linebreaks(html_content):
#     """
#     <body> 태그 기준으로:
#     1) script, style, noscript 제거
#     2) <br> → '\n', <p> 끝에도 '\n' 추가
#     3) id="content"만 추출
#     """
#     soup = BeautifulSoup(html_content, 'html.parser')

#     # 1) script, style, noscript 제거
#     for bad_tag in ["script", "style", "noscript"]:
#         for t in soup.find_all(bad_tag):
#             t.decompose()

#     # 2) id="content"만 추출
#     content_div = soup.find(id="content")
#     if not content_div:
#         return None

#     # <br>은 직접 '\n'으로 교체
#     for br in content_div.find_all("br"):
#         br.replace_with(" ")
    
#     # <p> 끝에는 '\n' 추가
#     for p in content_div.find_all("p"):
#         p.append(" ")

#     # 전체 텍스트
#     raw_text = content_div.get_text()

#     # 각 줄 단위로 나눈 뒤, 앞뒤 공백을 제거하고 내용이 있는 줄만 모음
#     lines = [line.strip() for line in raw_text.splitlines()]
#     # 내용이 없는 빈 줄은 제거
#     lines = [line for line in lines if line]

#     # 다시 여러 줄을 '\n'로 합침
#     final_text = " ".join(lines)

#     return final_text
def convert_html_to_text_with_linebreaks(html_content):
    """
    <body> 태그 기준으로:
    1) script, style, noscript 제거
    2) <br> → '\n', <p> 끝에도 '\n' 추가
    3) id="content"와 id="contents" 태그를 모두 추출
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # 1) script, style, noscript 제거
    for bad_tag in ["script", "style", "noscript"]:
        for t in soup.find_all(bad_tag):
            t.decompose()

    # 2) id="content"와 id="contents" 모두 추출
    content_divs = soup.find_all(id=lambda x: x in ["content", "contents"])

    if not content_divs:
        print("No 'content' or 'contents' id found.")
        return None

    # 텍스트를 모아 처리
    all_text = []
    for content_div in content_divs:

        for location_tag in content_div.find_all(class_="location"):
            location_tag.decompose()

        # <br>은 '\n'으로 교체
        for br in content_div.find_all("br"):
            br.replace_with(" ")
        
        # <p> 끝에는 '\n' 추가
        for p in content_div.find_all("p"):
            p.append(" ")
        
        # 각 태그의 텍스트를 추가
        raw_text = content_div.get_text()
        all_text.append(raw_text)

    # 여러 태그의 텍스트를 '\n'로 연결
    combined_text = " ".join(all_text)

    # 각 줄 단위로 나눈 뒤, 앞뒤 공백을 제거하고 내용이 있는 줄만 모음
    lines = [line.strip() for line in combined_text.splitlines()]
    # 내용이 없는 빈 줄은 제거
    lines = [line for line in lines if line]

    # 다시 여러 줄을 '\n'로 합침
    final_text = " ".join(lines)

    return final_text


def crawl_website(base_url, output_dir, min_depth=2, max_depth=4):
    """
    DFS로 전체 사이트 순회하면서 (min_depth 이상) id="content"만 텍스트를 저장한다.
    """
    to_visit = [(base_url, 0)]  # (URL, Depth) 목록

    while to_visit:
        current_url, depth = to_visit.pop()  # DFS: 맨 뒤 pop()

        if depth > max_depth:
            continue
        if current_url in visited:
            continue

        visited.add(current_url)
        print(f"Visiting: {current_url}, Depth: {depth}")

        try:
            response = requests.get(current_url, timeout=20)
            if response.status_code != 200:
                print(f"Failed to fetch {current_url}: {response.status_code}")
                continue

            html_content = response.text
            page_text = convert_html_to_text_with_linebreaks(html_content)

            if depth >= min_depth and page_text:
                page_data = {
                    "url": current_url,
                    "content": page_text
                }
                save_page_data(output_dir, current_url, page_data)
                print(f"Crawled and saved: {current_url}")

            # 하위 링크 추출 (DFS 계속)
            soup = BeautifulSoup(html_content, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                if not href:
                    continue
                next_url = urljoin(current_url, href)
                if is_same_domain(base_url, next_url) and next_url not in visited:
                    to_visit.append((next_url, depth + 1))

            time.sleep(1)

        except Exception as e:
            print(f"Error crawling {current_url}: {e}")




def is_same_domain(base_url, url):
    base_domain = urlparse(base_url).netloc
    target_domain = urlparse(url).netloc
    return base_domain == target_domain

def save_page_data(output_dir, url, page_data):
    parsed_url = urlparse(url)
    file_name = f"{parsed_url.netloc}{parsed_url.path}".replace("/", "_").strip("_") + ".json"
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(page_data, f, ensure_ascii=False, indent=4)
    print(f"Saved data to {file_path}")

def crawl_website(base_url, output_dir, min_depth=2, max_depth=4):
    """
    DFS로 전체 사이트 순회하면서 (min_depth 이상) 텍스트를 저장한다.
    """
    to_visit = [(base_url, 0)]  # (URL, Depth) 목록

    while to_visit:
        current_url, depth = to_visit.pop()  # DFS: 맨 뒤 pop()

        if depth > max_depth:
            continue
        if current_url in visited:
            continue

        visited.add(current_url)
        print(f"Visiting: {current_url}, Depth: {depth}")

        try:
            response = requests.get(current_url, timeout=20)
            if response.status_code != 200:
                print(f"Failed to fetch {current_url}: {response.status_code}")
                continue

            html_content = response.text
            page_text = convert_html_to_text_with_linebreaks(html_content)

            if depth >= min_depth and page_text:
                page_data = {
                    "url": current_url,
                    "content": page_text
                }
                save_page_data(output_dir, current_url, page_data)
                print(f"Crawled and saved: {current_url}")

            # 하위 링크 추출 (DFS 계속)
            soup = BeautifulSoup(html_content, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                if not href:
                    continue
                next_url = urljoin(current_url, href)
                if is_same_domain(base_url, next_url) and next_url not in visited:
                    to_visit.append((next_url, depth + 1))

            time.sleep(1)

        except Exception as e:
            print(f"Error crawling {current_url}: {e}")

if __name__ == "__main__":
    base_url = "https://www.yongin.go.kr"
    output_directory = "yongin_data2"
    os.makedirs(output_directory, exist_ok=True)

    crawl_website(base_url, output_directory, min_depth=2, max_depth=4)
    print(f"Crawling complete. Data saved to directory: {output_directory}")
