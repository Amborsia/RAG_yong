# models/embedding.py
import os
import openai
import numpy as np
from dotenv import load_dotenv
from tiktoken import encoding_for_model

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai

def calculate_tokens(text, model="text-embedding-3-small"):
    """
    주어진 텍스트의 토큰 수 계산
    """
    tokenizer = encoding_for_model(model)
    return len(tokenizer.encode(text))

def encode_texts(texts, batch_size=5):
    """
    OpenAI 'text-embedding-3-small' 모델을 사용하여 배치 형태로 텍스트를 임베딩.
    """
    if not texts:
        return np.array([], dtype=np.float32)

    embeddings = []
    total_texts = len(texts)

    for i in range(0, total_texts, batch_size):
        batch = texts[i:i + batch_size]
        try:
            # 배치 내 각 텍스트의 토큰 길이 확인 및 잘라내기(8191 토큰 제한 가정)
            for j, text in enumerate(batch):
                token_count = calculate_tokens(text, model="text-embedding-3-small")
                if token_count > 8191:
                    print(f"[Warning] Text {i+j} exceeds token limit ({token_count} tokens). Truncating.")
                    batch[j] = text[:1000]  # 임시로 앞부분만 사용

            # OpenAI API 호출
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            for item in response.data:
                embeddings.append(np.array(item.embedding, dtype=np.float32))

        except client.OpenAIError as e:
            print(f"[Error] OpenAI API call failed: {e}")
            embeddings.extend([np.zeros(1536, dtype=np.float32)] * len(batch))

        except Exception as e:
            print(f"[Error] Unexpected error: {e}")
            embeddings.extend([np.zeros(1536, dtype=np.float32)] * len(batch))

    return np.array(embeddings, dtype=np.float32)
