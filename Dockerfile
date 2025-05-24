# Digital Twin T1D SDK Docker Image
# Production-ready container for easy deployment

FROM python:3.9-slim

# Metadata
LABEL maintainer="Panos & Claude"
LABEL description="Digital Twin T1D SDK - Help 1 billion people with diabetes"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire SDK
COPY . .

# Install SDK in editable mode
RUN pip install -e .

# Create data directories
RUN mkdir -p /app/data/datasets /app/data/processed /app/logs

# Expose ports for REST API and dashboard
EXPOSE 8080 8081

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SDK_MODE=production
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sdk; print('SDK OK')" || exit 1

# Default command - start API server
CMD ["python", "-m", "sdk.api", "--host", "0.0.0.0", "--port", "8080"] 