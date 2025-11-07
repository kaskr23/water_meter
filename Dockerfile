FROM python:3.11-slim

# Install OS packages we need: Tesseract + libs for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Render sets PORT env variable; we just use it in app.py
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
