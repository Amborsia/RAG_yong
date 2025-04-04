"""Gemma LLM Wrapper"""

import os
from typing import Any, Dict, List, Optional, Union

import requests
from pydantic import BaseModel, Field

from ..custom_logging import LLMTracer


class GemmaMessage(BaseModel):
    """Gemma 메시지 모델"""

    role: str = Field(..., description="메시지 작성자의 역할 (system/user/assistant)")
    content: Union[str, List[Dict[str, str]]] = Field(..., description="메시지 내용")


class GemmaConfig(BaseModel):
    """Gemma 설정 모델"""

    model: str = Field(default="chat_model", description="사용할 모델명")
    max_tokens: int = Field(default=1000, description="최대 토큰 수")
    temperature: float = Field(default=0.7, description="온도 (0-1)")
    stream: bool = Field(default=True, description="스트리밍 여부")


class GemmaLLM:
    """Gemma LLM 래퍼"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[GemmaConfig] = None,
        project_name: Optional[str] = None,
    ):
        self.api_url = api_url or os.getenv("GEMMA_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("GEMMA_TOKEN")
        self.config = config or GemmaConfig()
        self.tracer = LLMTracer("gemma3")
        self.project_name = project_name

        if not self.api_key:
            raise ValueError("GEMMA_TOKEN이 설정되지 않았습니다.")

    def _format_messages(self, messages: List[GemmaMessage]) -> List[Dict[str, Any]]:
        """메시지 포맷팅"""
        formatted = []
        for msg in messages:
            if isinstance(msg.content, str):
                content = [{"type": "text", "text": msg.content}]
            else:
                content = msg.content
            formatted.append({"role": msg.role, "content": content})
        return formatted

    @property
    def trace(self):
        """추적 데코레이터"""
        return self.tracer.trace

    def chat(
        self, messages: List[GemmaMessage], stream: Optional[bool] = None
    ) -> Union[str, requests.Response]:
        """채팅 완성"""
        formatted_messages = self._format_messages(messages)

        # 사용자 질의와 컨텍스트 추출
        user_query = ""
        context = ""
        for msg in messages:
            if isinstance(msg.content, str):
                if msg.role == "user":
                    user_query = msg.content
                elif msg.role == "system":
                    if "교재 내용:" in msg.content:
                        # 교재 내용 부분 추출
                        parts = msg.content.split("교재 내용:")
                        if len(parts) > 1:
                            context = parts[1].split("학생의 질문:")[0].strip()

        trace_decorator = self.tracer.trace(project_name=self.project_name)

        @trace_decorator
        def _chat_impl(query: str) -> Dict[str, Any]:
            """실제 API 호출 및 응답 처리"""
            payload = {
                "model": self.config.model,
                "messages": formatted_messages,
                "max_tokens": self.config.max_tokens,
                "stream": stream if stream is not None else self.config.stream,
            }

            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                stream=payload["stream"],
            )
            response.raise_for_status()

            if payload["stream"]:
                return response

            result = response.json()["choices"][0]["message"]["content"]
            return {
                "query": query,
                "context": context or "No context provided",
                "response": result,
            }

        result = _chat_impl(user_query)

        # 스트리밍 응답은 그대로 반환
        if isinstance(result, requests.Response):
            return result

        # 일반 응답은 response 부분만 반환
        return result["response"]

    def stream(self, messages: List[GemmaMessage]) -> requests.Response:
        """스트리밍 채팅 완성"""
        return self.chat(messages, stream=True)

    @classmethod
    def from_env(cls, project_name: Optional[str] = None) -> "GemmaLLM":
        """환경 변수로부터 인스턴스 생성"""
        return cls(project_name=project_name)


# 편의를 위한 함수들
def create_message(role: str, content: str) -> GemmaMessage:
    """메시지 생성 헬퍼 함수"""
    return GemmaMessage(role=role, content=content)


def create_user_message(content: str) -> GemmaMessage:
    """사용자 메시지 생성"""
    return create_message("user", content)


def create_system_message(content: str) -> GemmaMessage:
    """시스템 메시지 생성"""
    return create_message("system", content)


def create_assistant_message(content: str) -> GemmaMessage:
    """어시스턴트 메시지 생성"""
    return create_message("assistant", content)
