# EBS 교재 기반 튜터링 챗봇 개발 계획

## 1. RAG 파이프라인 구축

### 1.1 벡터 저장소 구성

- VS2 API 호환성 유지
- 페이지 단위 벡터화 및 저장
- 현재 사용 중인 인덱스: `ebs-mini`
- 모델: `bge-m3`

### 1.2 검색 로직 개선

- VS2 API 응답의 pageNo를 활용한 JSON 조회
- 다중 페이지 검색 결과 처리
- 컨텍스트 통합 방식 구현

### API 호출 정보

```bash
# 환경변수 사용
POST ${VS2_URL}/${VS2_MODEL}/${INDEX_NAME}/\_search
{
  "sentences": user_query
}
```

### 응답 형식

```json
{
    "took": 93,
    "status": 200,
    "_model": "bge-m3",
    "_collection": "ebs-mini",
    "_vectors": null,
    "_objects": [
        {
            "_id": "c1dfd96e-ea8c-52b6-a785-275bca38ac26",
            "_score": 0.74684405,
            "_metadata": {
                "type": "declarative",
                "title": "뉴런과학1_미니북",
                "pageNo": "6",
                "_id": "6"
            }
        },
        ...
    ]
}
```

## 2. 프롬프트 엔지니어링

### 2.1 시스템 프롬프트 설계

- 교육용 튜터 페르소나 정의
- 학습 가이드 역할 정의
- 답변 형식 및 스타일 가이드라인

### 2.2 컨텍스트 구성

- 사용자 질문 + 검색 결과 통합 템플릿
- 다중 페이지 내용 통합 로직
- 교육적 맥락 유지 방안

## 3. 대화 흐름 설계

### 3.1 기본 질의-응답

- 교재 내용 기반 설명
- 예시와 함께 개념 설명
- 단계별 학습 가이드

### 3.2 심화 학습 기능

- 연관 개념 추천
- 문제 풀이 가이드
- 실생활 적용 예시 제시

## 4. 평가 및 개선

### 4.1 평가 지표

- 응답 정확도
- 교육적 효과성
- 사용자 만족도

### 4.2 개선 프로세스

- 사용자 피드백 수집
- 응답 품질 모니터링
- 정기적 성능 평가

## 5. 추가 기능 (향후 확장)

### 5.1 학습 진도 관리

- 학습 내용 기록
- 취약 부분 파악
- 맞춤형 학습 경로 제시

### 5.2 인터랙티브 요소

- 자동 퀴즈 생성
- 실습 가이드 제공
- 학습 동기 부여 요소

## 구현된 핵심 기능

### 1. VS2 검색 통합

```python
def search(query: str, top_k: int = 3):
    payload = {
        "sentences": [query]
    }
    # VS2 API 호출 및 결과 처리
```

### 2. 컨텍스트 통합

```python
context_chunks = []
for r in results:
    page_no = r["page_no"]
    content = r["content"]
    context_chunks.append(f"[{page_no}페이지]\n{content}")
```

### 3. 응답 생성

- GPT-4 스트리밍 응답
- 교육적 맥락 유지
- 출처 정보 표시

## 다음 단계 작업

1. PDF 뷰어 연동

   - PDF.js 또는 유사 라이브러리 검토
   - 페이지 동기화 구현
   - 하이라이트 기능 추가

2. 학습 기능 강화

   - 학습 진도 추적
   - 개념 이해도 평가
   - 복습 추천 시스템

3. 성능 최적화
   - 검색 결과 캐싱
   - 응답 생성 속도 개선
   - 메모리 사용량 최적화

## 설치 및 실행 방법

### 필요 조건

- Python 3.8 이상
- OpenAI API 키
- VS2 API 접근 권한

### 환경변수 설정

```bash
# .env
VS2_URL=your_vs2_url
VS2_MODEL=bge-m3
OPENAI_API_KEY=your_openai_api_key
```

### 설치

```bash
pip install streamlit python-dotenv requests langchain-openai
```

### 실행

```bash
streamlit run app.py
```

## 개발자 정보

이 EBS 교육용 튜터 챗봇은 VS2 벡터 검색과 GPT-4를 활용하여 중학 과학 학습을 지원합니다.
