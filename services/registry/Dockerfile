FROM python:3.12-slim
WORKDIR /app
COPY ml_platform ./ml_platform
COPY services/registry/app.py ./app.py
ENV PORT=8201
EXPOSE 8201
CMD ["python", "app.py"]
