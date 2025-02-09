# utils/chunking.py
from typing import List
from tiktoken import encoding_for_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def fixed_size_chunking(text: str, chunk_size: int = 500) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def token_based_chunking(text: str, max_tokens: int = 300, model: str = "gpt-3.5-turbo") -> List[str]:
    tokenizer = encoding_for_model(model)
    tokens = tokenizer.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

def recursive_chunking(text: str, max_tokens: int = 300, model: str = "gpt-3.5-turbo") -> List[str]:
    tokenizer = encoding_for_model(model)
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    
    mid_point = len(tokens) // 2
    left_text = tokenizer.decode(tokens[:mid_point])
    right_text = tokenizer.decode(tokens[mid_point:])
    return recursive_chunking(left_text, max_tokens, model) + recursive_chunking(right_text, max_tokens, model)

def sort_chunks_by_similarity(chunks: List[str], query: str) -> List[str]:
    """
    TF-IDF 기반으로 쿼리와 chunk 간 유사도를 계산해 정렬
    (chunk-level 임베딩 검색이 아니라 전통적 TF-IDF를 쓰고 싶을 때 사용)
    """
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query] + chunks)
    query_vector = vectors[0]
    chunk_vectors = vectors[1:]

    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    sorted_indices = similarities.argsort()[::-1]
    return [chunks[i] for i in sorted_indices]
