# ============================================================================
# Google Cloud Run 배포용 Dockerfile
# FAISS(C++ 바인딩) 및 FastAPI 비동기 스트리밍 서버 환경 구성
# ============================================================================
FROM python:3.11-slim

# 시스템 패키지 업데이트 및 FAISS 구동에 필요한 OpenMP 런타임(libgomp1) 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치 (Docker 레이어 캐싱 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY . .

# Cloud Run 기본 포트 환경변수 설정
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Uvicorn 서버 실행 (Cloud Run의 동적 PORT 바인딩 지원)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]

