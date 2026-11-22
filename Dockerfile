
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml uv.lock .python-version ./

# Install project dependencies without installing the project itself
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source and model
COPY src/ ./src/
COPY models/ ./models/

# Configure Streamlit
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Hugging Face Spaces default port
EXPOSE 7860

# Start Streamlit
CMD ["uv", "run", "--no-sync", "streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
