# Serving image for the Olympic Medal Predictor Streamlit app.
# Train first (python -m src.train) so models/model_bundle.joblib exists,
# then:
#   docker build -t olympic-medal-predictor .
#   docker run -p 8501:8501 --env-file .env olympic-medal-predictor
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY models/ models/
COPY pytest.ini .

EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
