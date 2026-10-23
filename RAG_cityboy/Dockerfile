Use official python slim image for lightweight deployment

FROM python:3.11-slim

Prevent Python from writing pyc files and enable unbuffered output for Cloud Run logs

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

Set working directory inside container

WORKDIR /app

Install system build dependencies required for C extensions if needed

RUN apt-get update && apt-get install -y --no-install-recommends 

build-essential 

curl 

&& rm -rf /var/lib/apt/lists/*

Copy dependency definition file

COPY requirements.txt .

Install dependencies without cache to minimize container layer size

RUN pip install --no-cache-dir -r requirements.txt

Copy application source code

COPY . /app

Expose default Cloud Run port

EXPOSE 8080

Run FastAPI app with Uvicorn server bound to 0.0.0.0:${PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
