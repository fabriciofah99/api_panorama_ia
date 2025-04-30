# Dockerfile com suporte ao FastAPI + OpenCV
FROM python:3.10-slim

# Instala dependências do sistema necessárias para OpenCV e Pillow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Copia os arquivos
COPY . .

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Expondo a porta da API
EXPOSE 8080

# Iniciando a aplicação FastAPI com Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
