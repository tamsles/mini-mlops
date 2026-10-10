
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY . .

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "mini_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
