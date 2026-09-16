# ==============================================================================
# STAGE 1: Dependency Builder Layer
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies required to compile high-performance math wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy python packaging blueprints to initialize system installs
COPY pyproject.toml requirements-lock.txt ./

# Build and isolate python package allocations inside a local wheelhouse
RUN pip install --no-cache-dir --user -r requirements-lock.txt

# ==============================================================================
# STAGE 2: Secure Production Runtime Instance Layer
# ==============================================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create a non-privileged system application user to manage execution isolation
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Copy installed python structures safely from the builder layer
COPY --from=builder /root/.local /home/appuser/.local
COPY . .

# Ensure the non-privileged system user owns the execution space
RUN chown -R appuser:appgroup /app

USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Expose internal interface gateway ports
EXPOSE 8000

# Fire the backend orchestration application tracking loop execution pipeline
CMD ["python", "interactive_gui.py"]
