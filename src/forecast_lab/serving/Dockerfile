FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn[standard] gunicorn
COPY . .
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "forecast_lab.serving.app:app"]