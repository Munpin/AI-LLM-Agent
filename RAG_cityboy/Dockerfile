# 1. 파이썬 경량 베이스 이미지 사용 (FROM 명령어 사용)
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 패키지 설치 (필요시)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. Cloud Run 환경 변수 PORT 지원 및 Gunicorn/Uvicorn 실행 설정
# Cloud Run은 환경 변수 PORT(기본 8080)로 트래픽을 전달합니다.
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
