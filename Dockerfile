FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md /app/
COPY ragtree /app/ragtree

RUN pip install --no-cache-dir -e ".[api]"

CMD ["ragtree", "doctor"]
