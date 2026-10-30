FROM python:3.12-slim

ENV UV_LINK_MODE=copy
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

ENV ORBITA_HOST=0.0.0.0 \
    ORBITA_DATA_DIR=/data \
    ORBITA_WHISPER_MODEL=small \
    ORBITA_WHISPER_DEVICE=cpu \
    ORBITA_WHISPER_COMPUTE_TYPE=int8
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uv", "run", "--no-dev", "orbita"]
