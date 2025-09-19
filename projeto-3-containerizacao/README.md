## 🐳 Objetivo
Desenvolver e containerizar uma API REST para predição de tipos de vidro usando Flask e Docker, demonstrando habilidades em deploy de modelos de machine learning em ambientes production-ready.

## 🛠️ Tecnologias Utilizadas
- **Python 3.9** e **Flask** - API Web
- **Docker** - Containerização
- **Scikit-learn** - Modelo de ML
- **Pandas** - Processamento de dados
- **Joblib** - Carregamento de modelos

## 🌐 Funcionalidades da API
- **Endpoint de saúde** (`/health`) - Verificação do status
- **Predição única** (`/predict`) - Predição para um único sample
- **Predição em lote** (`/batch_predict`) - Processamento de arquivos CSV
- **Documentação automática** - Endpoints auto-documentados
- **Gestão de erros** - Respostas apropriadas para diferentes erros

## 🎓 Aprendizados
1. **Desenvolvimento de APIs REST** com Flask
2. **Containerização** com Docker
3. **Deploy de modelos de ML** em produção
4. **Gestão de dependências** com requirements.txt
5. **Processamento de arquivos** em APIs web
6. **Boas práticas** de desenvolvimento web

## 🚀 Como Executar
docker run -p 5000:5000 vidro-prediction-api

### Construir a imagem Docker
```bash
cd projeto-3-containerizacao
docker build -t vidro-prediction-api .