import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# Configuração de estilo para os gráficos
plt.style.use('default')
sns.set_palette("deep")

def load_glass_data():
    """
    Carrega o dataset de identificação de vidros do UCI Repository
    """
    print("Carregando dataset de vidros...")
    
    # CORREÇÃO: Usar fetch_openml em vez de fetch_ucirepo
    glass = fetch_openml(name='glass', version=1, as_frame=True)
    
    # Criar DataFrame - estrutura diferente do fetch_openml
    df = glass.frame
    df.rename(columns={'class': 'Type'}, inplace=True)
    
    print(f"Dataset carregado com {df.shape[0]} amostras e {df.shape[1]} características")
    return df

def basic_statistics(df):
    """
    Gera estatísticas básicas do dataset
    """
    print("\n=== ESTATÍSTICAS BÁSICAS ===")
    print(df.describe())
    
    print("\n=== TIPOS DE VIDRO ===")
    print(df['Type'].value_counts().sort_index())
    
    return df.describe()

def plot_distributions(df):
    """
    Cria gráficos de distribuição para as variáveis
    """
    print("\nCriando gráficos de distribuição...")
    
    # Selecionar apenas colunas numéricas (excluir a coluna target 'Type')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'Type']
    
    # Configurar subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols[:9]):  # Limitar a 9 gráficos
        df[col].hist(bins=30, ax=axes[i])
        axes[i].set_title(f'Distribuição de {col}')
        axes[i].set_ylabel('Frequência')
    
    # Remover eixos vazios se houver menos de 9 colunas
    for i in range(len(numeric_cols), 9):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('distribuicoes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'distribuicoes.png'

def plot_correlations(df):
    """
    Cria heatmap de correlações
    """
    print("Criando heatmap de correlações...")
    
    # Calcular correlações apenas das features numéricas
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df[[col for col in numeric_df.columns if col != 'Type']]
    
    plt.figure(figsize=(12, 8))
    correlation_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, fmt='.2f')
    plt.title('Matriz de Correlação entre Variáveis')
    plt.tight_layout()
    plt.savefig('correlacoes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'correlacoes.png'

def plot_by_glass_type(df):
    """
    Cria gráficos agrupados por tipo de vidro
    """
    print("Criando gráficos por tipo de vidro...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # RI vs Tipo de Vidro
    df.boxplot(column='RI', by='Type', ax=axes[0, 0])
    axes[0, 0].set_title('Índice de Refração por Tipo de Vidro')
    axes[0, 0].set_ylabel('RI')
    
    # Na vs Tipo de Vidro
    df.boxplot(column='Na', by='Type', ax=axes[0, 1])
    axes[0, 1].set_title('Sódio (Na) por Tipo de Vidro')
    axes[0, 1].set_ylabel('Na')
    
    # Mg vs Tipo de Vidro
    df.boxplot(column='Mg', by='Type', ax=axes[1, 0])
    axes[1, 0].set_title('Magnésio (Mg) por Tipo de Vidro')
    axes[1, 0].set_ylabel('Mg')
    
    # Ca vs Tipo de Vidro
    df.boxplot(column='Ca', by='Type', ax=axes[1, 1])
    axes[1, 1].set_title('Cálcio (Ca) por Tipo de Vidro')
    axes[1, 1].set_ylabel('Ca')
    
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig('por_tipo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'por_tipo.png'

def main():
    """
    Função principal para executar toda a análise
    """
    print("Iniciando análise exploratória de dados de vidros...")
    
    # Carregar dados
    df = load_glass_data()
    
    # Mostrar informações básicas do dataset
    print("\n=== INFORMAÇÕES DO DATASET ===")
    print(df.info())
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    
    # Estatísticas básicas
    stats = basic_statistics(df)
    
    # Visualizações
    dist_plot = plot_distributions(df)
    corr_plot = plot_correlations(df)
    type_plot = plot_by_glass_type(df)
    
    print("\n=== ANÁLISE CONCLUÍDA ===")
    print("Gráficos salvos como:")
    print(f"- {dist_plot} (Distribuições)")
    print(f"- {corr_plot} (Correlações)")
    print(f"- {type_plot} (Por tipo de vidro)")
    
    return df, stats

if __name__ == "__main__":
    df, stats = main()