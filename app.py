# app.py
import streamlit as st
import pickle
import os

# 필요한 모듈들 import
import models.database as db
from initialize import init_rag
from services.generate import generate_answer

def main():
    st.title("Yongin RAG Demo")
    
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
            st.write("Re-building index from scratch...")
            init_rag(
                data_dir=data_dir,
                chunk_strategy=chunk_strategy,
                chunk_param=chunk_param,
                index_type=index_type,
                output_index_path="faiss_index.bin",
                output_chunk_path="chunked_data.pkl"
            )
            st.success("New index successfully built.")
        else:
            st.write("Loading existing index...")
            db.load_data(data_dir)
            db.load_index("faiss_index.bin", index_type=index_type)
            try:
                with open("chunked_data.pkl", "rb") as f:
                    db.chunked_data = pickle.load(f)
                st.success("Existing index loaded successfully.")
            except FileNotFoundError:
                st.error("chunked_data.pkl not found. Try disabling 'Use existing index'.")
            except Exception as e:
                st.error(f"Failed to load chunked_data: {e}")

    st.write("---")

    # 3) 질문 입력 및 답변 생성
    query = st.text_input("Ask a question about Yongin City:")
    if st.button("Generate Answer"):
        if not db.index:
            st.error("Index not initialized or loaded. Please initialize first.")
        else:
            answer = generate_answer(
                query=query,
                top_k=5,
                ranking_mode=ranking_mode,
                llm_backend=llm_backend
            )
            st.write("**Answer:**")
            st.write(answer)

if __name__ == "__main__":
    main()
