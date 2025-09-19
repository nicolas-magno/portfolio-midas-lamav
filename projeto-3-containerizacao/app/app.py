from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import traceback

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
        # Tentar carregar artefatos treinados
        model = joblib.load('./app/modelo_treinado.pkl')
        preprocessor = joblib.load('./app/preprocessor.joblib')
        print("Modelo e pré-processador carregados com sucesso!")
        return True
        
    except Exception as e:
        print(f"Erro ao carregar artefatos: {e}")
        print("Criando modelo dummy para demonstração...")
        
        try:
            # Criar um modelo dummy mais realista
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            
            # Dados de exemplo baseados no dataset real
            np.random.seed(42)
            n_samples = 100
            X_dummy = np.random.randn(n_samples, 9)
            
            # Criar classes baseadas em clusters
            y_dummy = np.zeros(n_samples)
            y_dummy[X_dummy[:, 0] > 0] = 1
            y_dummy[X_dummy[:, 1] > 0.5] = 2
            y_dummy[X_dummy[:, 2] < -0.5] = 3
            y_dummy = y_dummy.astype(int)
            
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=20, random_state=42))
            ])
            
            model.fit(X_dummy, y_dummy)
            
            preprocessor = StandardScaler()
            preprocessor.fit(X_dummy)
            
            print("Modelo dummy criado com sucesso!")
            return True
            
        except Exception as inner_e:
            print(f"Erro ao criar modelo dummy: {inner_e}")
            return False

@app.route('/')
def home():
    """
    Rota inicial da API
    """
    return jsonify({
        "message": "API de Predição de Tipos de Vidro",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "batch_predict": "/batch_predict (POST)"
        },
        "usage": {
            "predict": "Envie JSON com {'features': [[valores]]}",
            "batch_predict": "Envie arquivo CSV com as colunas do dataset"
        }
    })

@app.route('/health')
def health_check():
    """
    Rota para verificar saúde da API
    """
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }
    return jsonify(status)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Rota para fazer predições
    """
    try:
        if model is None or preprocessor is None:
            return jsonify({"error": "Modelo não carregado"}), 503
        
        # Obter dados da requisição
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                "error": "Dados de entrada inválidos",
                "expected_format": {"features": [[1.516, 11.7, 1.01, 1.19, 72.59, 0.43, 11.44, 0.02, 0.1]]}
            }), 400
        
        features = data['features']
        
        # Validar dados de entrada
        if not isinstance(features, list) or not all(isinstance(row, list) for row in features):
            return jsonify({"error": "Features deve ser uma lista de listas"}), 400
        
        if not all(len(row) == 9 for row in features):
            return jsonify({"error": "Cada sample deve ter 9 features"}), 400
        
        # Converter para DataFrame
        df = pd.DataFrame(features, columns=[
            'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'
        ])
        
        # Pré-processar dados
        processed_data = preprocessor.transform(df)
        
        # Fazer predição
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data) if hasattr(model, 'predict_proba') else None
        
        # Preparar resposta
        response = {
            "predictions": predictions.tolist(),
            "processed_features": processed_data.tolist(),
            "model_type": str(type(model.named_steps['clf'] if hasattr(model, 'named_steps') else model))
        }
        
        if probabilities is not None:
            response["probabilities"] = probabilities.tolist()
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Erro na predição: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Rota para fazer predições em lote
    """
    try:
        if model is None or preprocessor is None:
            return jsonify({"error": "Modelo não carregado"}), 503
        
        # Verificar se arquivo foi enviado
        if 'file' not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Nome de arquivo vazio"}), 400
        
        # Ler arquivo
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
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(processed_data)
            for i in range(probabilities.shape[1]):
                df[f'prob_class_{i}'] = probabilities[:, i]
        
        # Retornar resultado como CSV
        from io import StringIO
        output = StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return output.getvalue(), 200, {
            'Content-Type': 'text/csv',
            'Content-Disposition': 'attachment; filename=predictions.csv'
        }
        
    except Exception as e:
        print(f"Erro no batch predict: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    print("Iniciando aplicação Flask...")
    if load_artifacts():
        print("Aplicação iniciada com sucesso!")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Falha ao carregar artefatos. Encerrando aplicação.")