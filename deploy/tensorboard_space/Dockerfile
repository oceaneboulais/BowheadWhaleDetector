# Hugging Face Space (Docker SDK) that serves the training TensorBoard.
# HF Spaces route external traffic to $PORT (7860 by default).
FROM python:3.11-slim

RUN pip install --no-cache-dir tensorboard==2.14.0

WORKDIR /app
# TB event files for every run live under logs/<tag>/ (snapshot of ./runs).
COPY logs/ /app/logs/

EXPOSE 7860
CMD ["tensorboard", "--logdir", "/app/logs", \
     "--host", "0.0.0.0", "--port", "7860", \
     "--path_prefix", ""]
