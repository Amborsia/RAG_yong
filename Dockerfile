# 최신 Python 3.11 이미지 사용
FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# 필요 패키지 설치
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# 코드 복사
COPY . /app

# 서버 실행
CMD ["streamlit", "run", "app.py"]