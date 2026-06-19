# Hugging Face Space (Docker SDK) that serves the training TensorBoard.
# nginx proxies TensorBoard (internal :7861) on public :7860 and injects a
# one-line JS snippet that auto-navigates to the Embedding Projector tab
# (#projector) on first load.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends nginx && \
    rm -rf /var/lib/apt/lists/*

# numpy<2 because tensorboard 2.14 uses np.string_, removed in NumPy 2.0.
RUN pip install --no-cache-dir tensorboard==2.14.0 six "numpy<2"

WORKDIR /app
COPY logs/       /app/logs/
COPY nginx.conf  /etc/nginx/sites-enabled/default
COPY start.sh    /app/start.sh
RUN  chmod +x /app/start.sh && rm -f /etc/nginx/sites-enabled/default.bak

EXPOSE 7860
CMD ["/app/start.sh"]
