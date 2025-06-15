# ───────── BASE ─────────
FROM python:3.12-slim-bookworm

# ───────── OS DEPENDENCIES ─────────
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        build-essential gfortran libopenblas-dev liblapack-dev \
        libjpeg-dev libmagic1 poppler-utils tesseract-ocr git curl \
    && rm -rf /var/lib/apt/lists/*

# ───────── PYTHON DEPENDENCIES ─────────
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefer-binary -r requirements.txt \
 && pip install jupyterlab \
 && python -m ipykernel install --prefix /usr/local \
        --name "lab_ai_agent" \
        --display-name "Python (lab_ai_agent)"

# ───────── PROJECT SOURCE ─────────
COPY . .

# ───────── JUPYTER ─────────
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
