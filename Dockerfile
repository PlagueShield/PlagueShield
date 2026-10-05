FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAGUESHIELD_DATA_DIR=/data

WORKDIR /app
COPY pyproject.toml README.md ./
COPY plagueshield ./plagueshield
RUN pip install . && groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
COPY server ./server
COPY scripts/container_start.py ./scripts/container_start.py
COPY .deployment-seed /opt/plagueshield-seed

EXPOSE 8080
ENTRYPOINT ["python", "scripts/container_start.py"]
