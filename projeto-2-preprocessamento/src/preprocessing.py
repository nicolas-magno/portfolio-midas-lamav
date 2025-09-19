import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

class GlassDataPreprocessor:
    """
    Classe para pré-processamento de dados de vidros
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.numeric_features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
        self.preprocessor = None
        self.is_fitted = False
        
    def load_data(self, filepath=None):
        """
        Carrega os dados de vidros
        Se filepath for None, usa dataset do UCI
        """
        if filepath:
            print(f"Carregando dados de {filepath}")
            df = pd.read_csv(filepath)
        else:
            # CORREÇÃO: Usar fetch_openml em vez de fetch_ucirepo
            from sklearn.datasets import fetch_openml
            glass = fetch_openml(name='glass', version=1, as_frame=True)
            df = glass.frame
            df.rename(columns={'class': 'Type'}, inplace=True)
            
        print(f"Dados carregados: {df.shape[0]} amostras, {df.shape[1]} características")
        return df
    
    def create_preprocessor(self):
        """
        Cria o pipeline de pré-processamento
        """
        # Pipeline para features numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Pipeline para features categóricas (se houvessem)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combinar transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features)
                # Adicionar categóricas se necessário
            ])
        
        return self.preprocessor
    
    def split_data(self, df, target_col='Type'):
        """
        Divide os dados em treino e teste
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, 
            stratify=y  # Preserva proporção das classes
        )
        
        print(f"Divisão dos dados:")
        print(f"  Treino: {X_train.shape[0]} amostras")
        print(f"  Teste: {X_test.shape[0]} amostras")
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X_train):
        """
        Ajusta o pré-processador aos dados de treino
        """
        if self.preprocessor is None:
            self.create_preprocessor()
            
        print("Ajustando pré-processador...")
        self.preprocessor.fit(X_train)
        self.is_fitted = True
        print("Pré-processador ajustado!")
        
        return self
    
    def transform(self, X):
        """
        Aplica transformação aos dados
        """
        if not self.is_fitted:
            raise ValueError("Pré-processador não foi ajustado. Chame fit() primeiro.")
            
        print("Aplicando transformações...")
        X_transformed = self.preprocessor.transform(X)
        
        # Criar DataFrame com nomes das colunas
        feature_names = self.preprocessor.get_feature_names_out()
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
        
        return X_transformed_df
    
    def fit_transform(self, X_train):
        """
        Ajusta e transforma os dados
        """
        return self.fit(X_train).transform(X_train)
    
    def save_preprocessor(self, filepath='preprocessor.joblib'):
        """
        Salva o pré-processador treinado
        """
        if not self.is_fitted:
            raise ValueError("Pré-processador não foi ajustado. Nada para salvar.")
            
        joblib.dump(self.preprocessor, filepath)
        print(f"Pré-processador salvo em {filepath}")
    
    def load_preprocessor(self, filepath='preprocessor.joblib'):
        """
        Carrega um pré-processador salvo
        """
        self.preprocessor = joblib.load(filepath)
        self.is_fitted = True
        print(f"Pré-processador carregado de {filepath}")

def main():
    """
    Função principal para demonstrar o pré-processamento
    """
    print("=== DEMONSTRAÇÃO DO PRÉ-PROCESSAMENTO ===")
    
    # Inicializar pré-processador
    preprocessor = GlassDataPreprocessor()
    
    # Carregar dados
    df = preprocessor.load_data()
    
    # Dividir dados
    X_train, X_test, y_train, y_test = preprocessor.split_data(df)
    
    # Ajustar e transformar dados de treino
    print("\n--- Pré-processamento dos dados de treino ---")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transformar dados de teste
    print("\n--- Pré-processamento dos dados de teste ---")
    X_test_processed = preprocessor.transform(X_test)
    
    # Mostrar resultados
    print("\n=== RESULTADOS DO PRÉ-PROCESSAMENTO ===")
    print(f"Dados de treino processados: {X_train_processed.shape}")
    print(f"Dados de teste processados: {X_test_processed.shape}")
    
    print("\nPrimeiras 5 linhas dos dados processados:")
    print(X_train_processed.head())
    
    print("\nEstatísticas dos dados processados:")
    print(X_train_processed.describe())
    
    # Salvar pré-processador
    preprocessor.save_preprocessor()
    
    return preprocessor, X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test = main()