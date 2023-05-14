from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

def load_dataset():
    """
    Carrega o conjunto de dados Wine do Scikit-Learn.

    Retorna:
    X (np.array): Dados de entrada.
    y (np.array): Rótulos de destino.
    """
    X, y = load_wine(return_X_y = True)
    return X, y

def train_model(X : np.array, y : np.array) -> RandomForestClassifier:
    """
    Treina um modelo de floresta aleatória no conjunto de dados Wine.

    Parâmetros:
    X (np.array): Dados de entrada.
    y (np.array): Rótulos de destino.

    Retorna:
    model (RandomForestClassifier): O modelo treinado.
    """

    # Definir o modelo
    model = RandomForestClassifier(n_estimators = 2, max_depth = 1, random_state = 1)

    # Treinar o modelo
    model = model.fit(X, y)

    # Avaliar o treinamento do modelo
    print(f"Training Mean Accuracy: {model.score(X, y)}")

    return model

def save_model(model, filepath):
    """
    Salva o modelo treinado em um arquivo pickle.

    Parâmetros:
    model (qualquer): O modelo treinado para ser salvo.
    filepath (str): O caminho do arquivo onde o modelo treinado será salvo.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    """
    A função principal que executa as etapas de treinamento do modelo.

    Esta função carrega o conjunto de dados Titanic usando a função load_dataset(), 
    treina um modelo de floresta aleatória nos dados carregados usando a função train_model(),
    e retorna o modelo treinado.
    """
    X, y = load_dataset()
    model = train_model(X, y)
    save_model(model, 'model.pkl')
    
    return 


if __name__ == "__main__":
    main()
