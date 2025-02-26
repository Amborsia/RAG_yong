# init_langsmith.py
import os
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
import openai

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_openai_client():
    """
    LangSmith로 래핑된 OpenAI 클라이언트를 반환합니다.
    """
    return wrap_openai(openai)
