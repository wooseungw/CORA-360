FROM python:3.12-slim AS system

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    default-jre-headless \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cora

COPY requirements-repro.txt pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY reproduce.sh REPRODUCIBILITY.md MODEL_ZOO.md ./

FROM system AS source-check
RUN python -m compileall -q src scripts

FROM source-check AS runtime
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-repro.txt && \
    python -m pip install --no-deps .

RUN useradd --create-home --uid 10001 cora && chown -R cora:cora /opt/cora
USER cora

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import cora; print(cora.__version__)" || exit 1

ENTRYPOINT ["python", "scripts/inference.py"]
CMD ["--help"]
