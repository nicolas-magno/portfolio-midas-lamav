from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

# Inicializar aplicação Flask
app = Flask(__name__)

# Carregar modelo e pré-processador
model = None
preprocessor = None

def load_artifacts():
    """
    Carrega o modelo e pré-processador
    """
    global model, preprocessor
    
    try:
        # Em produção, esses arquivos estariam em um volume ou seriam
        # baixados durante a construção do container
        model = joblib.load('./app/modelo_treinado.pkl')
        preprocessor = joblib.load('./app/preprocessor.joblib')
        print("Modelo e pré-processador carregados com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar artefatos: {e}")
        # Criar um modelo dummy para demonstração se não encontrar os arquivos
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        print("Criando modelo dummy para demonstração...")
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Ajustar com dados dummy
        X_dummy = np.random.rand(10, 9)  # 9 features como no dataset de vidros
        y_dummy = np.random.randint(1, 3, 10)  # Classes 1, 2
        model.fit(X_dummy, y_dummy)
        
        preprocessor = StandardScaler()
        preprocessor.fit(X_dummy)

@app.route('/')
def home():
    """
    Rota inicial da API
    """
    return jsonify({
        "message": "API de Predição de Tipos de Vidro",
        "status": "online",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    })

@app.route('/health')
def health_check():
    """
    Rota para verificar saúde da API
    """
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Rota para fazer predições
    """
    try:
        # Obter dados da requisição
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({"error": "Dados de entrada inválidos. Use formato: {'features': [[...]]}"}), 400
        
        # Converter para DataFrame
        features = data['features']
        df = pd.DataFrame(features, columns=[
            'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'
        ])
        
        # Pré-processar dados
        processed_data = preprocessor.transform(df)
        
        # Fazer predição
        predictions = model.predict(processed_data)
        
        # Retornar resultado
        return jsonify({
            "predictions": predictions.tolist(),
            "processed_features": processed_data.tolist()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Rota para fazer predições em lote
    """
    try:
        # Verificar se arquivo foi enviado
        if 'file' not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Nome de arquivo vazio"}), 400
        
        # Ler arquivo CSV
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            return jsonify({"error": "Apenas arquivos CSV são suportados"}), 400
        
        # Verificar colunas
        expected_cols = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
        if not all(col in df.columns for col in expected_cols):
            return jsonify({
                "error": f"Colunas esperadas: {expected_cols}",
                "colunas_recebidas": df.columns.tolist()
            }), 400
        
        # Pré-processar dados
        processed_data = preprocessor.transform(df[expected_cols])
        
        # Fazer predição
        predictions = model.predict(processed_data)
        
        # Adicionar predições ao DataFrame
        df['prediction'] = predictions
        
        # Retornar resultado como CSV
        from io import StringIO
        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return output.getvalue(), 200, {'Content-Type': 'text/csv'}
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Iniciando aplicação Flask...")
    load_artifacts()
    app.run(host='0.0.0.0', port=5000, debug=True)
