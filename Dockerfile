FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src

RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000

# Default: serve the FastAPI surface; override with `ragtree doctor`, etc.
CMD ["ragtree", "serve", "--host", "0.0.0.0", "--port", "8000"]
