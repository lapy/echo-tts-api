# Echo-TTS API + minimal web UI — CUDA 12.6 on Ubuntu 24.04.
# Keep the image lean: Gradio and other dev-only paths are listed in `.dockerignore`.

FROM node:22-bookworm-slim AS webbuilder
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Pre-Ampere defaults: api_server uses ECHO_PRE_AMPERE=auto unless set (FP16 + low_mid when CC < 8.0).
# Volta (V100, sm_70): PyTorch must be CUDA 12.6 wheels (cu126). Official cu128/cu130 Linux
# binaries omit sm_70 — see https://github.com/pytorch/pytorch/blob/main/RELEASE.md
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    ECHO_PYTORCH_PIP_INDEX=cu126

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/echo-tts
ENV PATH="/opt/echo-tts/bin:$PATH"

WORKDIR /app

COPY requirements.txt requirements-torch.txt ./

# Install torch from cu126 only (Volta-safe). Using extra-index-url + PyPI can still pick a
# newer CUDA wheel without sm_70.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu126 -r requirements-torch.txt \
    && pip install -r requirements.txt

COPY . .
COPY --from=webbuilder /web/dist ./web/dist

EXPOSE 8000

CMD ["python", "api_server.py"]
