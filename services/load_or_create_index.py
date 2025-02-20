import os
import pickle

import streamlit as st

import models.database as db
from services.initialize import init_rag

DATA_DIR = "data/yongin_data2"
INDEX_FILE = "faiss_index.bin"
CHUNKED_FILE = "chunked_data.pkl"


#################################
# FAISS 인덱스 자동 생성 및 로드
#################################
def load_or_create_index():
    if os.path.exists(INDEX_FILE):
        db.load_data(DATA_DIR)
        db.load_index(INDEX_FILE, index_type="FLAT")
        try:
            with open(CHUNKED_FILE, "rb") as f:
                db.chunked_data = pickle.load(f)
        except FileNotFoundError:
            st.warning(
                "⚠️ `chunked_data.pkl` 파일이 없습니다. 일부 기능이 제한될 수 있습니다."
            )
        except Exception as e:
            st.error(f"❌ `chunked_data.pkl` 로드 중 오류 발생: {e}")
    else:
        st.write("🔄 FAISS 인덱스가 없습니다. 새로 생성 중...")
        init_rag(
            data_dir=DATA_DIR,
            chunk_strategy="recursive",
            chunk_param=500,
            index_type="FLAT",
            output_index_path=INDEX_FILE,
            output_chunk_path=CHUNKED_FILE,
        )
        st.success("✅ 새로운 FAISS 인덱스 생성 완료!")
        db.load_data(DATA_DIR)
        db.load_index(INDEX_FILE, index_type="FLAT")
        try:
            with open(CHUNKED_FILE, "rb") as f:
                db.chunked_data = pickle.load(f)
            st.success("✅ 인덱스 및 chunked_data 로드 완료!")
        except FileNotFoundError:
            st.warning("⚠️ `chunked_data.pkl` 파일이 여전히 없습니다.")
