FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
RUN useradd --create-home appuser
USER appuser
ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 75
