FROM python:3.12-slim
WORKDIR /app
COPY ml_platform ./ml_platform
COPY services/serving/app.py ./app.py
ENV PORT=8203
EXPOSE 8203
CMD ["python", "app.py"]
