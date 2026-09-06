# Stage 1: Build base environment and install dependencies
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime image for production
FROM python:3.10-slim AS runner

WORKDIR /app

# Copy python dependencies from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app/

# Copy application source code
COPY . /app

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Expose FastAPI port
EXPOSE 8001

# Run Uvicorn server production-ready
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
