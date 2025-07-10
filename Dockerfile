# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --upgrade pip && pip install -r requirements.txt \
    && apt-get remove -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip cache purge

# Copy project files
COPY . .

# Retain the .env file (ensure it is copied)
COPY .env .env

# Expose the port Chainlit/FastAPI runs on
EXPOSE 8000

# Default command to run Chainlit app (adjust if needed)
CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"] 
