# 배포 예시:
# gcloud run deploy seoul-rag --source . --region asia-northeast3 \
#   --allow-unauthenticated --memory 2Gi --max-instances 1 \
#   --set-env-vars GOOGLE_API_KEY=YOUR_KEY
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
