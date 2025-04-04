import logging
from pathlib import Path

# 기본 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 로그 디렉토리 설정
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
