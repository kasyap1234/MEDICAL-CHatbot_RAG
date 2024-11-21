# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend project
COPY . .

# Create a directory for data files
RUN mkdir -p /app/Data

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables for OpenAI and other configurations
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
