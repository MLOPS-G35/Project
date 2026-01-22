FROM ghcr.io/astral-sh/uv:python3.11-bookworm

WORKDIR /app

ENV WANDB_MODE=disabled

COPY pyproject.toml uv.lock ./

RUN sed -i '/^readme *= *"README.md"/d' pyproject.toml && \
    sed -i '/^license *= *{ *file *= *"LICENSE" *}/d' pyproject.toml

# Install only dependencies (not the project)
RUN uv sync --frozen --no-install-project

# Copy frontend code
COPY frontend ./frontend

EXPOSE 8501

CMD ["uv", "run", "--no-project", "streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
