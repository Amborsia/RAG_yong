import logging
import time

import langsmith as ls
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Ollama 관련 import
from langchain_community.llms import Ollama

from init_langsmith import get_openai_client
from services.search import search_top_k

load_dotenv()
openai_client = get_openai_client()

logging.basicConfig(
    filename="execution_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@ls.traceable(tags=["my-tag"], metadata={"query": "query", "top_k": "top_k"})
def generate_answer(
    query, top_k=5, ranking_mode="rrf", llm_backend="openai", stream=False
):
    rt = ls.get_current_run_tree()
    rt.metadata["query"] = query
    rt.metadata["top_k"] = top_k

    start_time = time.time()

    try:
        # 1) chunk-level 검색
        results = search_top_k(query, top_k=top_k, ranking_mode=ranking_mode)

        # ✅ 검색 결과가 없을 경우 대비
        if not results or len(results) == 0:
            return "❌ No relevant documents found. Please refine your query."

        rt.metadata["num_results"] = len(results)

        # ✅ top_k가 리스트 길이를 초과하지 않도록 제한
        top_k = min(len(results), top_k)

        # ✅ 리스트 접근 시 항상 범위 확인
        relevant_docs = results[:top_k]

        for i, r in enumerate(relevant_docs):
            doc_idx = r.get("doc_idx", "N/A")
            chunk_preview = r.get("chunk_text", "")[:200].replace("\n", "\\n")
            print(
                f"[Chunk {i}] doc_idx={doc_idx}, chunk_text starts with: {chunk_preview}..."
            )

        # 2) 검색된 chunk를 합쳐 context 구성
        answer_chunks = []
        for r in relevant_docs[:3]:  # 최대 3개만 사용
            chunk_text = r.get("chunk_text", "내용 없음")
            doc_url = r.get("original_doc", {}).get("url", "출처 없음")
            enriched_chunk = (
                f"이 chunk는 {doc_url} 에서 가져온 내용입니다.\n{chunk_text}"
            )
            answer_chunks.append(enriched_chunk)

        context_text = "\n\n".join(answer_chunks)

        # 3) messages 구성
        messages = [
            {
                "role": "system",
                "content": (
                    """당신은 용인시청에 특화된 챗봇 역할을 맡고 있습니다.
                    오직 용인시와 관련된 주제, 문서에 대해서만 답변할 수 있습니다.
                    만약 사용자의 질문이 용인시와 무관하거나, 주어진 chunk에 관련 정보가 없다면
                    '해당 내용에 대한 정보가 없습니다'라고 답해 주세요.
                    아래의 규칙에 맞춰서 답변해 주세요:
                    1. 먼저 '질문 : [사용자 질문]'을 명시해 주세요.
                    2. 짧은 인사말이나 요약을 한두 줄 정도 작성해 주세요.
                    3. '대상', '내용', '방식', '주의사항', '사용처', '신청방법' 등
                    필요한 항목들을 bullet point(ー)로 나누어 안내해 주세요.
                    4. 추가 문의, 담당부서 등 정보를 항목 별로 구분해서 작성해 주세요.
                    5. 절대 새로운 URL(https://...)을 임의로 만들지 마세요.
                    6. chunk 내용에 존재하는 URL, 혹은 용인시나 정부 공식 웹사이트(예: yongin.go.kr, gov.kr)만 사용하세요.
                    7. 마지막에는 '감사합니다.'라는 문장과 함께 참고자료 링크를 알려주세요.
                    답변 시, 위 규칙에 맞춰 꼭 구조화된 형태로 요약하여 한국어로 제공해 주세요."""
                ),
            },
            {
                "role": "user",
                "content": f"""아래 chunk들을 참고하여 질문에 답해 주세요:

{context_text}

질문: {query}
""",
            },
        ]

        # 4) LLM 호출 분기
        answer = "❌ LLM 호출 실패"  # 기본값

        if llm_backend == "openai":
            from langchain_openai import ChatOpenAI

            try:
                # ChatOpenAI를 스트리밍 여부에 따라 생성 (streaming 매개변수 전달)
                llm = ChatOpenAI(
                    model_name="gpt-4o-mini", temperature=0.1, streaming=stream
                )
                if stream:
                    # LangChain이 제공하는 streaming 인터페이스 사용
                    response = llm.stream_chat(messages)
                    answer = ""
                    for chunk in response:
                        # chunk가 ChatMessage 객체이거나 dict인 경우 처리
                        token = (
                            chunk.content
                            if hasattr(chunk, "content")
                            else chunk.get("content", "")
                        )
                        answer += token
                        yield token
                    rt.metadata["openai_answer"] = answer
                else:
                    response = llm(messages)
                    answer = response.content.strip()
                    rt.metadata["openai_answer"] = answer
            except Exception as e:
                print(f"[Error] OpenAI API call failed: {e}")
                rt.metadata["openai_error"] = str(e)

        elif llm_backend == "ollama_deepseek":
            try:
                # (A) messages를 하나의 대형 문자열로 합침
                system_msg = messages[0]["content"]
                user_msg = messages[1]["content"]
                combined_prompt = f"{system_msg}\n\n{user_msg}"

                # (B) Ollama + DeepSeek R1 로컬 LLM 사용
                llm = Ollama(model="deepseek-r1:8b")  # or 'deepseek-r1:1.5b'
                chain = LLMChain(
                    llm=llm, prompt=PromptTemplate.from_template("{input}")
                )

                # (C) 실행
                answer = chain.run(combined_prompt)
                print("이것이 대답입니다. ->")
                print(answer)
                rt.metadata["ollama_answer"] = answer
            except Exception as e:
                print(f"[Error] Ollama API call failed: {e}")
                rt.metadata["ollama_error"] = str(e)

        else:
            raise ValueError(f"Unknown llm_backend={llm_backend}")

    except Exception as e:
        print(f"[Error] An error occurred: {e}")
        answer = "❌ 시스템 오류 발생"
        rt.metadata["error"] = str(e)

    end_time = time.time()
    rt.metadata["processing_time"] = end_time - start_time
    print(f"Answer generated in {rt.metadata['processing_time']:.2f} seconds.")

    logging.info(
        f"Query: {query}, Top_k: {top_k}, RankingMode: {ranking_mode}, LLM: {llm_backend}, "
        f"Time: {rt.metadata['processing_time']:.2f}s, Answer: {answer[:100]}..."
    )

    if stream:
        # 예시: 전체 답변을 단어 단위로 스트리밍 (실제 스트리밍 API 사용 시 해당 방식에 맞게 수정)
        for word in answer.split():
            yield word + " "
    else:
        return answer
