# 파이썬 3.11 경량 베이스 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /lib/apt/lists/*

# requirements.txt 복사 및 파이썬 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY main.py .

# Cloud Run 기본 포트인 8080 노출
EXPOSE 8080

# Uvicorn 서버 실행 (Gunicorn/Uvicorn worker 설정)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
