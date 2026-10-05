FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir opencv-python-headless numpy requests
# Optional real detector: pip install ultralytics  (then run agent with --detector yolo)
COPY simulator ./simulator
COPY vision-engine ./vision-engine
