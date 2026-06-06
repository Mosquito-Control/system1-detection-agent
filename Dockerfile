FROM python:3.11-slim

WORKDIR /app

# OpenCV headless needs these libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY system1/ system1/
COPY cameras.yaml .

ENV PYTHONUNBUFFERED=1
ENV OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp

CMD ["python", "-m", "system1.main"]
