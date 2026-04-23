# ---------------------------------------------------------------------------
# Pest Detection Pipeline - Hugging Face Spaces (Docker SDK) image
#
# Scope: runs the Flask app with the Model Inference tab + Kitchen Generator
# tab only. The Test / Real Video Generator tabs stay hidden via CLOUD_MODE=1
# because their Metric3D / Blender-ish dependencies are NOT installed here
# (keeps the image under ~2 GB).
#
# Build:      docker build -t pest-detector .
# Run local:  docker run -p 7860:7860 -e GEMINI_API_KEY=... -v $(pwd)/checkpoints:/data pest-detector
# Deploy:     push to huggingface.co/spaces/<you>/synthetic-pest-gen
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# System deps:
#   ffmpeg          - MP4 encode/decode in OpenCV
#   libgl1, libsm6, libxext6 - OpenCV runtime libs
#   curl, ca-certificates    - CHECKPOINT_URL downloads
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libsm6 \
        libxext6 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer). CPU-only torch keeps the image
# small: the full CUDA wheel is >2 GB.
COPY training/requirements.txt /app/training/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
         -r /app/training/requirements.txt \
    && pip install --no-cache-dir \
         "flask>=3.0" \
         "python-dotenv>=1.0" \
         "google-generativeai>=0.7" \
         "huggingface-hub>=0.24" \
         "requests>=2.31"

# Application source. Copying last maximizes cache hits on dep changes.
COPY . /app

# HF Spaces injects PORT=7860 but being explicit helps local `docker run`.
ENV CLOUD_MODE=1 \
    PORT=7860 \
    HF_DATA_DIR=/data

# Ensure the persistent volume directory exists even without a mount so the
# storage shim can create outputs/ under /app on cold container runs.
RUN mkdir -p /data /app/outputs

EXPOSE 7860

# HF Spaces expects PID 1 to be the web server. app/main.py already binds to
# $PORT when present.
CMD ["python", "-m", "app.main"]
