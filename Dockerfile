# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for audio and document processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directory for models/cache with full permissions
RUN mkdir -p /.cache && chmod -R 777 /.cache

# Set environment variables for memory efficiency
ENV MKL_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV TRANSFORMERS_CACHE=/.cache
ENV HF_HOME=/.cache

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app", "--timeout", "120", "--workers", "1", "--threads", "4"]
