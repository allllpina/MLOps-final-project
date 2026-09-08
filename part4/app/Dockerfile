FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
      fastapi \
      uvicorn \
      prometheus-fastapi-instrumentator \
      scikit-learn \
      joblib \
      numpy

COPY app.py model.pkl ./

EXPOSE 8085
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8085"]
