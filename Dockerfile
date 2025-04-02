# Dockerfile com suporte ao lama-cleaner e FastAPI
FROM python:3.10-slim

# Instala dependências do sistema necessárias para OpenCV e SAM
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /app

# Copia os arquivos do projeto
COPY . .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Expõe as portas
EXPOSE 8080

# Comando para rodar FastAPI + LamaCleaner (simples)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8080"]