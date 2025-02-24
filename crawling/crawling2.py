import json
import os
from time import sleep
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "https://www.yongin.go.kr/common/orgcht/BD_orgcht.do?q_domainCode=1"

def crawl_all_departments():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    wait = WebDriverWait(driver, 10)
    driver.get(BASE_URL)
    try:
        dept_link_elements = wait.until(EC.presence_of_all_elements_located(
            (By.XPATH, "//a[contains(@onclick, 'moveDeptGuide')]")
        ))
    except Exception as e:
        print("부서 링크 요소 로드 실패:", e)
        driver.quit()
        return

    dept_names = []
    for elem in dept_link_elements:
        text = elem.text.strip()
        if text and text not in dept_names:
            dept_names.append(text)
    print("찾은 부서:", dept_names)

    documents = []  # Document 객체 리스트 생성

    for dept in dept_names:
        try:
            driver.get(BASE_URL)
            dept_link = wait.until(EC.element_to_be_clickable(
                (By.XPATH, f"//a[contains(@onclick, 'moveDeptGuide') and normalize-space(text())='{dept}']")
            ))
            print(f"'{dept}' 부서 클릭")
            dept_link.click()
        except Exception as e:
            print(f"부서 '{dept}' 링크 클릭 실패: {e}")
            continue

        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody")))
        except Exception as e:
            print(f"부서 '{dept}'의 테이블 로딩 대기 실패: {e}")
            continue

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            print(f"부서 '{dept}'의 테이블을 찾지 못했습니다.")
            continue

        tbody = table.find("tbody")
        if not tbody:
            print(f"부서 '{dept}'의 <tbody>를 찾지 못했습니다.")
            continue

        rows = tbody.find_all("tr")
        dept_entries = []
        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue
            row_data = [cell.get_text(strip=True) for cell in cells]
            # 예: "자치분권과장 / 031-6193-2120 / 자치분권과 업무 총괄"
            dept_entries.append(" / ".join(row_data))
        print(f"부서 '{dept}' 데이터 수집 완료: {len(dept_entries)} 행")

        if dept_entries:
            # 부서별 Document 하나로 생성
            document_text = f"{dept} 부서 정보:\n" + "\n".join(dept_entries)
            doc = {
                "text": document_text,
                "metadata": {"department": dept}
            }
            documents.append(doc)
        sleep(0.5)

    driver.quit()

    output_path = "departments_documents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)
    print(f"크롤링 완료. Document 데이터가 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    os.chdir("output")
    crawl_all_departments()