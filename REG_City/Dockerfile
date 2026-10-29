# =====================================================================
# Dockerfile - 도시 정책 RAG 서버 (Google Cloud Run)
# ---------------------------------------------------------------------
# [빌드 & 배포 예시]
#   gcloud run deploy city-policy-rag \
#     --source . \
#     --region asia-northeast3 \
#     --allow-unauthenticated \
#     --memory 1Gi --cpu 1 --timeout 600 \
#     --max-instances 1 \
#     --no-cpu-throttling \
#     --set-secrets GOOGLE_API_KEY=GOOGLE_API_KEY:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,SUPABASE_SERVICE_ROLE_KEY=SUPABASE_SERVICE_ROLE_KEY:latest,SERVER_API_KEY=SERVER_API_KEY:latest \
#     --set-env-vars SUPABASE_URL=https://xxxx.supabase.co,LLM_MODEL=gemini-3.5-flash-lite
#   ※ --max-instances 1     : 작업 상태를 메모리에 보관하므로 단일 인스턴스 권장
#   ※ --no-cpu-throttling   : 응답 반환 후에도 백그라운드 스레드가 CPU 를 쓰도록
#   ※ API 키는 Secret Manager 에 저장하고 --set-secrets 로 주입하는 것을 권장
# =====================================================================

# 가벼운 공식 Python 이미지
FROM python:3.12-slim

# 파이썬 바이트코드 미생성, 로그 즉시 출력(버퍼링 해제 → Cloud Logging 에 바로 표시)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# 의존성 먼저 설치 → 소스만 바뀌면 이 레이어는 캐시 재사용
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 애플리케이션 소스 복사
COPY main.py .

# 보안: root 가 아닌 일반 사용자로 실행
RUN useradd --create-home --uid 1001 appuser
USER appuser

EXPOSE 8080

# 워커 1개: 작업(job) 상태를 프로세스 메모리에서 공유하기 위함
# Cloud Run 이 주입하는 $PORT 를 사용
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75
