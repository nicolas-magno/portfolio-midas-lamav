## ⚙️ Objetivo
Desenvolver um sistema robusto e automatizado para pré-processamento de dados de vidros, preparando-os para algoritmos de machine learning com técnicas adequadas de limpeza, transformação e normalização.

## 🛠️ Tecnologias Utilizadas
- **Python 3.9**
- **Scikit-learn** - Pipelines e transformers
- **Pandas** - Manipulação de dados
- **Joblib** - Serialização de modelos
- **Scikit-learn** - Validação cruzada

## 🔧 Funcionalidades
- **Carregamento flexível** de dados (UCI ou arquivos locais)
- **Imputação de valores missing** com estratégia mediana
- **Normalização** com StandardScaler
- **Divisão estratificada** treino/teste
- **Serialização** do pré-processador treinado
- **Validação cruzada** integrada

## 🎓 Aprendizados
1. **Design de pipelines** com Scikit-learn
2. **Imputação de valores missing** de forma apropriada
3. **Normalização** de features numéricas
4. **Validação cruzada estratificada** para dados desbalanceados
5. **Serialização** de objetos Python com Joblib
6. **Programação orientada a objetos** para sistemas de ML

## 🚀 Como Executar
cd projeto-2-preprocessamento
python src/preprocessing.py

### Pré-requisitos
```bash
pip install scikit-learn pandas joblib