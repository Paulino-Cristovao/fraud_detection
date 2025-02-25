# Use official Python slim image
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install production dependencies
RUN poetry install --no-dev --no-interaction --no-root

# Run the training script
CMD ["poetry", "run", "python", "train.py"]
