import json
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# 1. JSON 데이터 로드 (예: "departments_data.json")
with open("crawling/output/departments_documents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []
for dept, entries in data.items():
    for entry in entries:
        # entry 예시: ["시민소통관", "031-6193-2040", "시민소통관 업무 총괄"]
        # 부서명을 포함하여 각 항목을 보기 좋게 포맷팅합니다.
        content = f"부서: {dept}\n" + "\n".join(entry)
        documents.append(Document(page_content=content))

# 2. 임베딩 모델 초기화 (여기서는 OpenAI 임베딩 사용, API 키 등은 사전에 설정)
embeddings = OpenAIEmbeddings()

# 3. FAISS 벡터 스토어 생성 (문서로부터 인덱스 생성)
vectorstore = FAISS.from_documents(documents, embeddings)

# 4. 인덱스 저장 (추후 로드하여 RAG 시스템에 활용 가능)
vectorstore.save_local("rag_index")
print("RAG 인덱스 생성 및 저장 완료")