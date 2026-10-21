# The decision server on Linux, with CLM on PyTorch (backend = "torch" under [clm]).
#
#   docker build -t clm-decision .
#   docker run --rm clm-decision                                   # the fast tests
#   docker run --rm -m 28g -v clm-hf:/cache -e CLM_TEST_BACKEND=torch clm-decision uv run pytest -m slow -k real_server
#
# Models are downloaded into /cache; mount a volume there to keep them between runs. With an NVIDIA GPU and
# the NVIDIA container toolkit, add --gpus all and CLM runs on the GPU.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv HF_HOME=/cache
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .

CMD ["uv", "run", "pytest", "-q"]
