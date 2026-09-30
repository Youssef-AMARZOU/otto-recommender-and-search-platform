FROM python:3.12-slim

WORKDIR /app

# libgomp: OpenMP runtime for lightgbm / faiss on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "otto_rec.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
