import streamlit as st
import pickle
import os
import models.database as db
from initialize import init_rag
from services.generate import generate_answer

# ✅ FAISS 인덱스 파일 경로
INDEX_FILE = "faiss_index.bin"
CHUNKED_FILE = "chunked_data.pkl"
DATA_DIR = "data/yongin_data2"

# ✅ FAISS 인덱스 자동 생성 및 로드
def load_or_create_index():
    if os.path.exists(INDEX_FILE):
        st.write("📥 기존 FAISS 인덱스 로드 중...")
        db.load_index(INDEX_FILE, index_type="FLAT")

        try:
            with open(CHUNKED_FILE, "rb") as f:
                db.chunked_data = pickle.load(f)
            st.success("✅ FAISS 인덱스 로드 완료!")
        except FileNotFoundError:
            st.warning("⚠️ `chunked_data.pkl` 파일이 없습니다. 일부 기능이 제한될 수 있습니다.")
        except Exception as e:
            st.error(f"❌ `chunked_data.pkl` 로드 중 오류 발생: {e}")
    else:
        st.write("🔄 FAISS 인덱스가 없습니다. 새로 생성 중...")
        init_rag(
            data_dir=DATA_DIR,
            chunk_strategy="recursive",  # 기본 설정
            chunk_param=500,
            index_type="FLAT",
            output_index_path=INDEX_FILE,
            output_chunk_path=CHUNKED_FILE
        )
        st.success("✅ 새로운 FAISS 인덱스 생성 완료!")

        # 생성된 인덱스 로드
        db.load_index(INDEX_FILE, index_type="FLAT")
        try:
            with open(CHUNKED_FILE, "rb") as f:
                db.chunked_data = pickle.load(f)
            st.success("✅ 인덱스 로드 완료!")
        except FileNotFoundError:
            st.warning("⚠️ `chunked_data.pkl` 파일이 여전히 없습니다.")

# ✅ Streamlit 실행 시 FAISS 인덱스 자동 생성
st.write("🚀 FAISS 인덱스 확인 중...")
load_or_create_index()

def main():
    st.title("Yongin RAG Demo")
    
    query = st.text_input("Ask about Yongin City:")
    if st.button("Generate Answer"):
        if not db.index or db.index.ntotal == 0:
            st.error("❌ FAISS Index not initialized. Please restart the app.")
        else:
            answer = generate_answer(
                query=query,
                top_k=5,
                ranking_mode="rrf",
                llm_backend="openai"
            )
            st.write("**Answer:**")
            st.write(answer)

if __name__ == "__main__":
    main()
