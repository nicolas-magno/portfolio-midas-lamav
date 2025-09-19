# Usar imagem base do Python
FROM python:3.9-slim

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos de requisitos
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY app/ ./app/

# Expor a porta que a aplicação vai usar
EXPOSE 5000

# Comando para rodar a aplicação
CMD ["python", "./app/app.py"]
