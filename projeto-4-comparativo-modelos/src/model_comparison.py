import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, 
    classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import time
import joblib

# Importar modelos
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class ModelComparator:
    """
    Classe para comparar o desempenho de diferentes modelos de ML
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def initialize_models(self):
        """
        Inicializa os modelos a serem comparados
        """
        print("Inicializando modelos...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'SVM': SVC(
                random_state=self.random_state, probability=True
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB()
        }
        
        return self.models
    
    def load_data(self, X_train, X_test, y_train, y_test):
        """
        Carrega os dados para treino e teste
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Dados carregados:")
        print(f"  Treino: {X_train.shape[0]} amostras")
        print(f"  Teste: {X_test.shape[0]} amostras")
        print(f"  Número de classes: {len(np.unique(y_train))}")
        
        return self
    
    def cross_validate_models(self, cv=5):
        """
        Executa validação cruzada para todos os modelos
        """
        print("\n=== EXECUTANDO VALIDAÇÃO CRUZADA ===")
        
        results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            start_time = time.time()
            
            print(f"Validando {name}...")
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=skf, scoring='accuracy', n_jobs=-1
            )
            
            end_time = time.time()
            
            results[name] = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': end_time - start_time
            }
            
            print(f"  Acurácia: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"  Tempo de treino: {end_time - start_time:.2f}s")
        
        self.results = results
        return results
    
    def train_and_evaluate_models(self):
        """
        Treina e avalia todos os modelos no conjunto de teste
        """
        print("\n=== TREINANDO E AVALIANDO MODELOS ===")
        
        test_results = {}
        
        for name, model in self.models.items():
            start_time = time.time()
            
            print(f"Treinando {name}...")
            # Treinar modelo
            model.fit(self.X_train, self.y_train)
            
            # Fazer predições
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test) if hasattr(model, "predict_proba") else None
            
            end_time = time.time()
            
            # Calcular métricas
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            test_results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'training_time': end_time - start_time,
                'model': model,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
            print(f"  Acurácia no teste: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Tempo de treino: {end_time - start_time:.2f}s")
        
        # Identificar melhor modelo
        best_model_name = max(test_results.items(), key=lambda x: x[1]['accuracy'])[0]
        self.best_model = test_results[best_model_name]['model']
        
        print(f"\nMelhor modelo: {best_model_name} "
              f"(Acurácia: {test_results[best_model_name]['accuracy']:.4f})")
        
        # Adicionar resultados ao dicionário principal
        for name in self.results:
            if name in test_results:
                self.results[name].update(test_results[name])
        
        return test_results
    
    def plot_comparison(self):
        """
        Cria gráficos comparando o desempenho dos modelos
        """
        print("\nGerando gráficos de comparação...")
        
        # Extrair dados para plotagem
        model_names = list(self.results.keys())
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        test_accuracies = [self.results[name]['accuracy'] for name in model_names]
        training_times = [self.results[name]['training_time'] for name in model_names]
        
        # Gráfico de comparação de acurácia
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Acurácia (CV vs Teste)
        x = np.arange(len(model_names))
        width = 0.35
        
        ax1.bar(x - width/2, cv_means, width, label='Validação Cruzada', alpha=0.8)
        ax1.bar(x + width/2, test_accuracies, width, label='Conjunto de Teste', alpha=0.8)
        
        ax1.set_xlabel('Modelos')
        ax1.set_ylabel('Acurácia')
        ax1.set_title('Comparação de Acurácia entre Modelos')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tempo de treinamento
        ax2.bar(model_names, training_times, alpha=0.8, color='orange')
        ax2.set_xlabel('Modelos')
        ax2.set_ylabel('Tempo (segundos)')
        ax2.set_title('Tempo de Treinamento dos Modelos')
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparacao_modelos.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico de matriz de confusão para o melhor modelo
        best_model_name = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        y_pred = self.results[best_model_name]['y_pred']
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {best_model_name}')
        plt.ylabel('Verdadeiro')
        plt.xlabel('Predito')
        plt.tight_layout()
        plt.savefig('matriz_confusao.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return 'comparacao_modelos.png', 'matriz_confusao.png'
    
    def save_results(self, filename='resultados_modelos.csv'):
        """
        Salva os resultados em um arquivo CSV
        """
        results_df = pd.DataFrame.from_dict(self.results, orient='index')
        results_df = results_df[['cv_mean', 'cv_std', 'accuracy', 'f1_score', 'training_time']]
        results_df.columns = ['Validação Cruzada (Média)', 'Validação Cruzada (Desvio)', 
                             'Acurácia (Teste)', 'F1-Score (Teste)', 'Tempo de Treino (s)']
        results_df = results_df.round(4)
        results_df.to_csv(filename)
        
        print(f"Resultados salvos em {filename}")
        return results_df
    
    def save_best_model(self, filename='melhor_modelo.joblib'):
        """
        Salva o melhor modelo em disco
        """
        if self.best_model is not None:
            joblib.dump(self.best_model, filename)
            print(f"Melhor modelo salvo em {filename}")
        else:
            print("Nenhum modelo treinado para salvar")

def main():
    """
    Função principal para executar a comparação de modelos
    """
    print("=== COMPARAÇÃO DE MODELOS DE MACHINE LEARNING ===")
    
    # Carregar dados (usando o pré-processador do projeto 2)
    try:
        # Tentar importar do projeto anterior
        import sys
        sys.path.append('../projeto-2-preprocessamento/src')
        from preprocessing import GlassDataPreprocessor
        
        preprocessor = GlassDataPreprocessor()
        df = preprocessor.load_data()
        X_train, X_test, y_train, y_test = preprocessor.split_data(df)
        
        # Ajustar e transformar dados
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        print("Usando dados de exemplo para demonstração...")
        
        # Gerar dados de exemplo
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000, n_features=9, n_classes=5, n_informative=7,
            random_state=42
        )
        X_train_processed, X_test_processed, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    # Inicializar comparador
    comparator = ModelComparator()
    comparator.load_data(X_train_processed, X_test_processed, y_train, y_test)
    comparator.initialize_models()
    
    # Executar validação cruzada
    cv_results = comparator.cross_validate_models()
    
    # Treinar e avaliar modelos
    test_results = comparator.train_and_evaluate_models()
    
    # Gerar gráficos
    plot1, plot2 = comparator.plot_comparison()
    
    # Salvar resultados
    results_df = comparator.save_results()
    comparator.save_best_model()
    
    print("\n=== COMPARAÇÃO CONCLUÍDA ===")
    print("Resultados:")
    print(results_df)
    print(f"\nGráficos salvos como:")
    print(f"- {plot1} (Comparação de modelos)")
    print(f"- {plot2} (Matriz de confusão)")
    
    return comparator

if __name__ == "__main__":
    comparator = main()
