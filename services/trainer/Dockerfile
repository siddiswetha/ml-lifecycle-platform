FROM python:3.12-slim
WORKDIR /app
COPY ml_platform ./ml_platform
COPY services/trainer/app.py ./app.py
ENV PORT=8202
EXPOSE 8202
CMD ["python", "app.py"]
