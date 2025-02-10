# RAG (Retrieval-Augmented Generation) Project

이 프로젝트는 Faiss를 사용한 문서 검색과 OpenAI GPT 모델을 결합해 질문에 답변해주는 RAG 파이프라인을 구현한다. Nest.js나 Node.js가 아닌 FastAPI와 Python을 사용하며, 문서를 여러 chunk로 나누어 검색 정확도를 높였다.

## 목차

- 프로젝트 구조
- 설치 및 환경 구성
- 실행 방법
- 사용 예시
- Chunking 및 인덱스 종류 변경
- FAQ
- 주의사항
- 라이선스

# 프로젝트 구조

```
.
├── main.py                # FastAPI 서버 실행
├── initialize.py          # RAG 인덱스 생성(문서 로드→임베딩→FAISS 인덱스)
├── app.py                 # Streamlit 데모 UI (Yongin RAG Demo)
├── models/
│   ├── database.py        # FAISS 인덱스, 문서 로딩/저장
│   └── embedding.py       # OpenAI Embedding API
├── services/
│   ├── generate.py        # 최종 답변 생성 (검색→GPT 요약)
│   └── search.py          # 검색 로직 (임베딩→FAISS)
├── utils/
│   └── chunking.py        # 다양한 chunk 분할 방법 (token/fixed/recursive)
├── requirements.txt       # Python 의존성 목록 (faiss-cpu/faiss-gpu 등)
└── ...
```

- **main.py**: FastAPI 서버 구동, `/ask` 엔드포인트에서 질문 처리
- **initialize.py**: 문서를 JSON에서 읽어 chunk로 나눈 뒤 임베딩 & 인덱스 생성 후 저장
- **app.py**: Streamlit을 이용한 간단한 웹 UI 제공. Yongin RAG Demo
- **models**: DB 관리(`database.py`), OpenAI 임베딩(`embedding.py`)
- **services**: 검색(`search.py`), 답변 생성(`generate.py`) 등 RAG 주요 로직
- **utils/chunking.py**: 문서 chunking 전략(토큰 기반, 고정 길이, 재귀 분할)

# 설치 및 환경 구성

### Python 3.9+ (가급적 3.9.16 이상 추천)

### 가상환경(venv 또는 Conda) 사용 권장

```bash
# 가상환경 생성(예: venv)
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

- **faiss-cpu 또는 faiss-gpu 중 하나가 설치되어 있어야 한다.**
  - macOS 등에서 설치가 어려울 시 Conda를 사용:
    ```bash
    conda install faiss-cpu -c conda-forge
    ```

### OpenAI API 키

루트 경로에 `.env` 파일을 만들고, `OPENAI_API_KEY=sk-...` 형태로 API 키를 지정해야 함.

```bash
# .env 예시
OPENAI_API_KEY=sk-abc123abc123...
```

# 실행 방법

### streamlit, 서버 두개 동시 실행!

### 1) streamlit 켜기

```bash
streamlit run app.py
```

- streamlit을 통해 UI에서 파라미터를 확인 가능

### 2) 서버 실행

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

- 서버가 `127.0.0.1:8000`에서 기동
- 초기화 시점에 `faiss_index.bin`, `chunked_data.pkl`를 로딩해 검색 준비 완료

### 3) 질문 요청 (streamlit)

- 브라우저에서 http://localhost:8501로 접속 후,
- Initialize RAG 버튼을 눌러 인덱스를 새로 빌드하거나 로드
- 질문(예: "용인시 관광명소 알려줘")을 입력 후 Generate Answer 클릭

# 사용 예시

### 단일 질문

- 질문: "용인시 모현읍 이름유래가 뭐야?"
- `/ask`로 POST 요청 → FAISS 검색 → GPT 요약 → 응답

### Chunk 전략 바꿔보기

- `initialize.py`에서 `token_based_chunking` 대신 `fixed_size_chunking` 함수를 사용하도록 변경
- 다시 실행:
  ```bash
  python initialize.py  # faiss_index.bin 갱신
  python main.py
  ```

# Chunking 및 인덱스 종류 변경

### Chunking

- `utils/chunking.py`에 `fixed_size_chunking`, `token_based_chunking`, `recursive_chunking` 함수들이 존재
- `initialize.py`에서 원하는 함수 호출 가능:
  ```python
  token_based_chunking(combined_text, max_tokens=500)
  ```

### FAISS 인덱스

- `models/database.py`의 `build_index`에서 **"FLAT"** vs. **"HNSW"** 선택 가능
  - `index_type="HNSW"`로 설정 시 HNSW 인덱스 생성
  - `index_type="FLAT"`이면 IndexFlatL2 방식 사용

# FAQ

### "ModuleNotFoundError: No module named 'faiss'"

- `pip install faiss-cpu` (또는 `faiss-gpu`)가 설치되지 않은 경우.
- macOS 등에서 설치가 안 되면 Conda 사용 권장:
  ```bash
  conda install faiss-cpu -c conda-forge
  ```

### 오류 메시지: "No relevant chunks found."

- 검색 결과가 없거나, JSON 문서가 없을 때 발생.
- 문서에 해당 내용이 전혀 없을 수 있으니 확인하거나, chunk 크기/전처리를 조정해본다.

### OpenAI API 호환 문제

- `.env`에서 `OPENAI_API_KEY`가 없으면 임베딩 or GPT 호출 단계에서 실패
- API 호출이 너무 빈번하면 rate limit이 걸릴 수 있음.

### GPU 사용

- `faiss-gpu` 설치 후, `models/database.py`에서 `index_cpu_to_gpu`를 추가로 설정해야 한다(주석 처리된 예시 참고).

# 주의사항

- **OpenAI API 비용**: Embedding, ChatCompletion 호출 시 과금이 될 수 있으니 실제 운영 환경에서는 호출 횟수를 관리해야 함.
- **데이터 기밀**: JSON 문서에 민감 정보가 포함되어 있다면, 외부 노출 주의.
- **Chunk 크기**: 너무 크게 잡으면 토큰 제한 초과 가능, 너무 작으면 검색 정확도는 올라가지만 인덱스가 커져 검색 속도가 느려질 수 있음.
- **토큰 수 제한**: `embedding.py`에서 8191 토큰을 기준으로 잘라내는 로직이 있음. 필요한 경우 수정 가능.
- **에러 핸들링**: "No relevant docs" 등 상황에서 GPT를 호출하지 않고 사용자에게 "자료 없음"만 반환하도록 커스터마이징 가능.

# 라이선스

본 프로젝트는 MIT License 하에 공개될 수 있음

- OpenAI API 및 FAISS 라이브러리는 각각의 라이선스를 따른다.
- 상세 내용은 LICENSE 파일 및 각 라이브러리 라이선스 참조

# 전체 구성도(요약)

1. **initialize.py**
   - JSON 로드 → 청킹( `chunk_strategy`, `chunk_param` ) → 임베딩 → FAISS 인덱스( `index_type` )
   - 결과를 `faiss_index.bin`, `chunked_data.pkl` 저장
2. **main.py**
   - 서버 시작 시점에 `use_existing_index`로 인덱스 재생성 여부 결정
   - `ranking_mode`( `dense`, `tfidf`, `rrf` ), `llm_backend`( `openai`, `ollama_deepseek` ) 등도 지정 가능
   - `/ask` 엔드포인트가 `generate_answer(..., ranking_mode, llm_backend)` 호출
3. **services/search.py**
   - `search_top_k`에서 Dense + Sparse 검색 후, RRF 등으로 합산
   - `ranking_mode`에 따라 **Dense만** / **TF-IDF만** / **둘 결합(RRF)**
4. **services/generate.py**
   - 검색된 chunk를 맥락으로, `llm_backend="openai"`면 OpenAI GPT, `"ollama_deepseek"`면 Ollama(DeepSeek) 호출
   - “한국어로만 답해라 / 추론 노출 금지” 등 시스템 프롬프트로 후처리

---

# 최종 파라미터 표

| 파라미터               | 값 범위                             | 설명                                                         |
| ---------------------- | ----------------------------------- | ------------------------------------------------------------ |
| **use_existing_index** | `True` / `False`                    | 기존 인덱스/청크를 로드할지, 새로 빌드할지                   |
| **chunk_strategy**     | `"fixed"`, `"token"`, `"recursive"` | 문서를 어떻게 나눌지 (문자 기반 / 토큰 기반 / 재귀)          |
| **chunk_param**        | 정수(예: 300, 500, 800)             | 청킹 크기 (토큰 or 문자), `max_tokens=...`, `chunk_size=...` |
| **index_type**         | `"FLAT"`, `"HNSW"`                  | FAISS 인덱스 종류 (Dense 검색용)                             |
| **ranking_mode**       | `"dense"`, `"tfidf"`, `"rrf"`       | 검색 랭킹 방식: Dense만, Sparse만, 또는 Hybrid(RRF)          |
| **llm_backend**        | `"openai"`, `"ollama_deepseek"`     | 최종 LLM 호출: OpenAI GPT vs 로컬(DeepSeek R1)               |
