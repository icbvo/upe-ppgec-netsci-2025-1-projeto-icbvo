# Dockerfile para o projeto:
# Link Prediction in Collaboration Networks using GNNs

# 1) Imagem base: Python 3.11 slim
FROM python:3.11-slim

# 2) Variáveis de ambiente básicas
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# 3) Instalar dependências de sistema mínimas
#    (compilador, git e libs úteis para algumas wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 4) Definir diretório de trabalho dentro do container
WORKDIR /workspace

# 5) Copiar apenas requirements primeiro (melhora cache do Docker)
COPY requirements.txt ./requirements.txt

# 6) Atualizar pip e instalar dependências Python
#    Incluo jupyter e networkx caso não estejam no requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install jupyter networkx

# 7) Copiar o restante do projeto
COPY . .

# 8) Criar pasta de resultados (se ainda não existir)
RUN mkdir -p results

# 9) Expor porta do Jupyter
EXPOSE 8888

# 10) Comando padrão:
#     - Sobe um Jupyter Lab apontando para /workspace
#     - Você pode sobrescrever esse CMD para rodar o treino, se quiser
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=", "--notebook-dir=/workspace"]
