FROM python:3.12-slim

# Instala o uv copiando da imagem oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copia os arquivos de dependência primeiro (otimiza o cache do Docker)
COPY pyproject.toml uv.lock ./

# Instala as dependências
RUN uv sync --frozen --no-dev --no-cache

# Copia o restante do código
COPY . .

# Expõe a porta que o FastAPI vai usar
EXPOSE 8000

# Comando para rodar o servidor usando o venv criado pelo uv
CMD ["/app/.venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]