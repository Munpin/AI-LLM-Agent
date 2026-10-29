# ------------------------------------------------------------
# 현진건 RAG 서버 - Google Cloud Run 배포용 Dockerfile
# ------------------------------------------------------------
FROM python:3.11-slim

# 파이썬 로그 즉시 출력 / .pyc 생성 안 함 / Chroma 익명 통계 끄기
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ANONYMIZED_TELEMETRY=False \
    CHROMA_DIR=/tmp/chroma_db \
    PORT=8080

WORKDIR /app

# chromadb 일부 의존성 빌드를 대비한 최소 빌드 도구
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 설치 → 소스만 바뀌면 이 레이어는 캐시 재사용
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8080

# job 상태를 메모리에 두므로 worker 는 반드시 1개
# (Cloud Run 이 주입하는 $PORT 를 쓰기 위해 shell 형식 사용)
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75
