# Use the official Python image with GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the application files into the container
COPY . /app/

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Expose the port that the FastAPI app will run on
EXPOSE 8080

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
