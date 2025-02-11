import streamlit as st
import pickle
import os

# 필요한 모듈들 import
import models.database as db
from initialize import init_rag
from services.generate import generate_answer

# 디버그 모드 설정 (True: 디버그 모드, False: 사용자 모드)
mode = st.sidebar.radio("Select Mode", ["User Mode", "Debug Mode"])

# ✅ 실행 시 기존 인덱스를 자동 로드하거나, 없으면 새로 생성하는 함수
def load_or_create_index():
    index_file = "faiss_index.bin"
    chunked_file = "chunked_data.pkl"
    data_dir = "data/yongin_data2"  # 기본 데이터 디렉토리

    if os.path.exists(index_file):
        st.write("📥 Auto-loading existing FAISS index...")
        db.load_index(index_file, index_type="FLAT")  # 기본적으로 FLAT 사용

        try:
            with open(chunked_file, "rb") as f:
                db.chunked_data = pickle.load(f)
            st.success("✅ Existing FAISS index loaded successfully!")
        except FileNotFoundError:
            st.warning("⚠️ chunked_data.pkl not found. Some features may not work.")
        except Exception as e:
            st.error(f"❌ Failed to load chunked_data: {e}")
    else:
        st.write("🔄 FAISS index not found. Creating a new index...")
        init_rag(
            data_dir=data_dir,
            chunk_strategy="recursive",  # 기본값 설정
            chunk_param=500,
            index_type="FLAT",
            output_index_path=index_file,
            output_chunk_path=chunked_file
        )
        st.success("✅ New FAISS index successfully created!")

        # 생성한 인덱스 로드
        db.load_index(index_file, index_type="FLAT")
        try:
            with open(chunked_file, "rb") as f:
                db.chunked_data = pickle.load(f)
            st.success("✅ Index creation and loading complete!")
        except FileNotFoundError:
            st.warning("⚠️ chunked_data.pkl still missing. Search may be limited.")

# **User Mode일 경우 실행 시 인덱스를 자동 로드 or 생성**
if mode == "User Mode":
    load_or_create_index()

def main():
    st.title("Yongin RAG Demo")

    if mode == "Debug Mode":
        st.subheader("🛠 Debug Mode (Advanced Settings)")

        # 1) 파라미터 입력 부
        data_dir = st.text_input("Data Directory", value="data/yongin_data2")
        chunk_strategy = st.selectbox("Chunk Strategy", ["fixed", "token", "recursive"], index=1)
        chunk_param = st.number_input("Chunk Param", min_value=1, value=500)
        index_type = st.selectbox("Index Type", ["FLAT", "HNSW"], index=1)
        ranking_mode = st.selectbox("Ranking Mode", ["dense", "tfidf", "rrf"], index=2)
        llm_backend = st.selectbox("LLM Backend", ["openai", "ollama_deepseek"], index=0)
        use_existing_index = st.checkbox("Use existing index", value=True)

        st.write("---")

        # 2) 인덱스 초기화(혹은 기존 인덱스 로드)
        if st.button("Initialize RAG"):
            if not use_existing_index:
                st.write("🔄 Re-building index from scratch...")
                init_rag(
                    data_dir=data_dir,
                    chunk_strategy=chunk_strategy,
                    chunk_param=chunk_param,
                    index_type=index_type,
                    output_index_path="faiss_index.bin",
                    output_chunk_path="chunked_data.pkl"
                )
                st.success("✅ New index successfully built.")
            else:
                load_or_create_index()

        st.write("---")

    # 3) 질문 입력 및 답변 생성 (모든 모드에서 공통)
    st.subheader("🔍 Ask a Question")

    query = st.text_input("Ask about Yongin City:")
    if st.button("Generate Answer"):
        if not db.index or db.index.ntotal == 0:
            st.error("❌ FAISS Index not initialized. Please initialize in Debug Mode.")
        else:
            answer = generate_answer(
                query=query,
                top_k=5,
                ranking_mode="rrf" if mode == "User Mode" else ranking_mode,
                llm_backend="openai" if mode == "User Mode" else llm_backend
            )
            st.write("**Answer:**")
            st.write(answer)

if __name__ == "__main__":
    main()
