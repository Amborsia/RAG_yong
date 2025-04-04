"""LLM 추적 시스템"""

import json
import logging
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class LLMTracer:
    """범용 LLM 추적 시스템"""

    def __init__(
        self,
        model_name: str,
        log_dir: Union[str, Path] = "logs",
        max_content_length: int = 1000,
    ):
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.max_content_length = max_content_length

        # 로그 디렉토리 생성
        self.log_dir.mkdir(exist_ok=True)

        # 로거 설정
        self.logger = logging.getLogger(f"llm_tracer_{model_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # 상위 로거로 전파 방지

        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 파일 핸들러 설정
        file_handler = logging.FileHandler(
            self.log_dir / f"{model_name}_traces.jsonl", encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(file_handler)

    def _truncate_text(self, text: str) -> str:
        """텍스트를 최대 길이로 제한"""
        if len(text) > self.max_content_length:
            return text[: self.max_content_length] + "..."
        return text

    def _estimate_tokens(self, text: str) -> int:
        """간단한 토큰 수 추정 (공백 기준)"""
        return len(text.split())

    def trace(
        self, project_name: Optional[str] = None, tags: Optional[List[str]] = None
    ):
        """LLM 호출 추적 데코레이터"""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                trace_id = f"{self.model_name}-{int(time.time() * 1000)}"
                start_time = time.time()

                # 입력값 캡처
                input_text = args[0] if args else kwargs.get("prompt", "")

                # 기본 태그 설정
                default_tags = [self.model_name, "chat"]
                if project_name:
                    default_tags.append(project_name)
                if tags:
                    default_tags.extend(tags)

                # 기본 메타데이터 설정
                metadata = {
                    "trace_id": trace_id,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_name,
                    "project": project_name,
                    "tags": default_tags,
                    "start_time": start_time,
                }

                try:
                    # 함수 실행
                    result = func(*args, **kwargs)
                    end_time = time.time()

                    # 구조화된 결과 처리
                    if isinstance(result, dict) and all(
                        k in result for k in ["query", "context", "response"]
                    ):
                        metadata.update(
                            {
                                "input": {
                                    "query": self._truncate_text(result["query"]),
                                    "context": self._truncate_text(result["context"]),
                                    "length": len(result["query"])
                                    + len(result["context"]),
                                },
                                "output": {
                                    "text": self._truncate_text(result["response"]),
                                    "length": len(result["response"]),
                                },
                            }
                        )
                        result_text = result["response"]
                    else:
                        # 기존 방식 유지 (하위 호환성)
                        result_text = str(result) if result is not None else ""
                        metadata.update(
                            {
                                "input": {
                                    "text": self._truncate_text(input_text),
                                    "length": len(input_text),
                                },
                                "output": {
                                    "text": self._truncate_text(result_text),
                                    "length": len(result_text),
                                },
                            }
                        )

                    # 성공 메타데이터 추가
                    metadata.update(
                        {
                            "status": "success",
                            "end_time": end_time,
                            "duration": round(end_time - start_time, 3),
                            "metrics": {
                                "tokens_prompt": self._estimate_tokens(input_text),
                                "tokens_completion": self._estimate_tokens(result_text),
                                "tokens_total": self._estimate_tokens(input_text)
                                + self._estimate_tokens(result_text),
                            },
                        }
                    )

                except Exception as e:
                    end_time = time.time()
                    # 실패 메타데이터 추가
                    metadata.update(
                        {
                            "status": "error",
                            "end_time": end_time,
                            "duration": round(end_time - start_time, 3),
                            "error": {
                                "type": type(e).__name__,
                                "message": str(e),
                                "traceback": str(e.__traceback__),
                            },
                        }
                    )
                    raise

                finally:
                    # JSONL 형식으로 로깅
                    self.logger.info(json.dumps(metadata, ensure_ascii=False))

                return result

            return wrapper

        return decorator

    def read_traces(
        self,
        n: int = 10,
        status: Optional[str] = None,
        project: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """트레이스 로그 읽기"""
        traces = []
        log_file = self.log_dir / f"{self.model_name}_traces.jsonl"

        if not log_file.exists():
            return traces

        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    trace = json.loads(line.strip())

                    # 필터링 조건 적용
                    if status and trace.get("status") != status:
                        continue
                    if project and trace.get("project") != project:
                        continue
                    if start_time and trace.get("timestamp") < start_time:
                        continue
                    if end_time and trace.get("timestamp") > end_time:
                        continue
                    if tags and not all(tag in trace.get("tags", []) for tag in tags):
                        continue

                    traces.append(trace)
                except json.JSONDecodeError:
                    continue

        return traces[-n:]

    def get_statistics(
        self,
        project: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """트레이스 통계 정보"""
        traces = self.read_traces(
            n=1000, project=project, start_time=start_time, end_time=end_time, tags=tags
        )

        total_calls = len(traces)
        successful_calls = sum(1 for t in traces if t.get("status") == "success")
        error_calls = sum(1 for t in traces if t.get("status") == "error")
        avg_duration = (
            sum(t.get("duration", 0) for t in traces) / total_calls
            if total_calls > 0
            else 0
        )

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "error_calls": error_calls,
            "success_rate": (
                (successful_calls / total_calls * 100) if total_calls > 0 else 0
            ),
            "avg_duration": round(avg_duration, 3),
            "total_tokens": sum(
                t.get("metrics", {}).get("tokens_total", 0) for t in traces
            ),
            "time_range": {
                "start": min((t.get("timestamp") for t in traces), default=None),
                "end": max((t.get("timestamp") for t in traces), default=None),
            },
        }


# Gemma3 트레이서 인스턴스 생성
gemma_tracer = LLMTracer("gemma3")

# Gemma3 관련 함수들
gemma_trace = gemma_tracer.trace
read_gemma_traces = gemma_tracer.read_traces
get_trace_statistics = gemma_tracer.get_statistics
