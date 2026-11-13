FROM python:3.14-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

ENV PATH="/code/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /code
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY predict.py model.bin ./

EXPOSE 9696
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
