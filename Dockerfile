FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# opencv-python-headless 에 필요한 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU 전용 PyTorch 먼저 설치 (CUDA 버전 대비 ~1.5GB 절약)
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성 (torch/torchvision은 이미 설치되어 있으므로 스킵됨)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
