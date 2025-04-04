# EBS 중학 과학 학습 도우미 챗봇 🔬

EBS 중학 과학 교재를 기반으로 학생들의 학습을 돕는 AI 챗봇 서비스입니다. 교재 내용을 바탕으로 학생들의 질문에 답변하고, 관련 페이지를 찾아 보여주는 기능을 제공합니다.

## 주요 기능 🌟

- **교재 기반 질의응답**: EBS 중학 과학 교재 내용을 기반으로 학생들의 질문에 답변
- **페이지 참조**: 답변과 관련된 교재 페이지를 자동으로 찾아 제시
- **PDF 뷰어**: 교재 페이지를 이미지로 변환하여 바로 확인 가능
- **대화 기록**: 이전 질문과 답변 내역을 사이드바에서 확인 가능

## 기술 스택 💻

### 벡터 데이터베이스

- **OpenQueryVS**: 교재 내용 임베딩 및 검색
  - 효율적인 의미 기반 검색
  - 실시간 쿼리 처리 최적화

### LLM (Large Language Model)

브랜치에 따라 두 가지 LLM 옵션을 제공합니다:

1. **OpenAI GPT-4** (`ebs-rag` 브랜치)

   - 강력한 자연어 이해 및 생성 능력
   - 스트리밍 응답 지원
   - API 키 필요

2. **Gemma3** (`gemma3` 브랜치)
   - 오픈소스 기반 로컬 실행 가능

### 프론트엔드

- **Streamlit**: 대화형 UI 구현
  - PDF 뷰어 통합
  - 실시간 스트리밍 응답
  - 반응형 사이드바 구현

## 시스템 아키텍처 🏗️

```mermaid
graph LR
    A[PDF 교재] --> B[Synap OCR]
    B --> C[텍스트 추출]
    D[사용자 질문] --> E[OpenQueryVS]
    C --> E
    E --> F[사용자 질의 + 컨텍스트]
    F --> G[LLM<br/>(OpenAI/Gemma3)]
    G --> H[답변 생성]
    H --> I[Streamlit UI]
```

## 설치 및 실행 🚀

1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

2. 환경 변수 설정

```bash
cp .env.example .env
# OpenAI 사용 시 API 키 설정 필요
# Gemma3 사용 시 별도 설정 불필요
```

3. 브랜치 선택

```bash
# OpenAI GPT-4 사용 시
git checkout ebs-rag

# Gemma3 사용 시
git checkout gemma3
```

4. 실행

```bash
streamlit run app.py
```

## 프로젝트 구조 📁

```
.
├── app.py                # Streamlit 애플리케이션 진입점
├── main.py              # FastAPI 서버
├── data/
│   └── ebs/            # EBS 교재 데이터
│       ├── chunks/     # 청킹된 JSON 파일
│       ├── pdfs/       # 원본 PDF 파일
│       └── texts/      # 추출된 텍스트 파일
├── models/
│   ├── database.py     # 데이터베이스 관리
│   └── embedding.py    # 임베딩 처리
├── services/
│   ├── ebs.py         # EBS RAG 핵심 로직
│   ├── handlers/      # 요청 처리기
│   ├── ui/           # UI 컴포넌트
│   │   ├── pdf_modal.py
│   │   ├── sidebar.py
│   │   └── styles.py
│   └── state/        # 상태 관리
├── utils/
│   ├── chat.py       # 채팅 유틸리티
│   ├── chunking.py   # 텍스트 청킹
│   └── prompts.py    # 프롬프트 관리
└── prompts/          # 프롬프트 템플릿
    └── ebs_tutor.yaml
```

## 라이선스 📝

MIT License

## 기여하기 🤝

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
