import logging


def setup_file_logger(log_file="debug.log", level=logging.DEBUG):
    """
    지정된 파일에 로그를 기록하도록 로깅 설정을 구성합니다.
    이미 핸들러가 설정되어 있다면 중복 추가를 방지합니다.
    """
    logger = logging.getLogger("rag_logger")
    logger.setLevel(level)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


# 전역 로거 객체 생성
logger = setup_file_logger()


def log_debug(message: str):
    """
    주어진 메시지를 디버그 레벨로 파일에 기록합니다.
    """
    logger.debug(message)
