from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_dataset():
    """
    Carrega o conjunto de dados Titanic do Scikit-Learn.

    Retorna:
    X (DataFrame): Dados de entrada.
    y (Series): Rótulos de destino.
    """
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    return X, y

def train_model(X, y):
    """
    Treina um modelo de floresta aleatória no conjunto de dados Titanic.

    Parâmetros:
    X (DataFrame): Dados de entrada.
    y (Series): Rótulos de destino.

    Retorna:
    model (RandomForestClassifier): O modelo treinado.
    """
    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pré-processamento
    numeric_features = ['age', 'fare']
    categorical_features = ['embarked', 'sex', 'pclass']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Definir o modelo
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])

    # Treinar o modelo
    model.fit(X_train, y_train)

    # Avaliar o modelo
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    return model

def main():
    """
    A função principal que executa as etapas de treinamento do modelo.

    Esta função carrega o conjunto de dados Titanic usando a função load_dataset(), 
    treina um modelo de floresta aleatória nos dados carregados usando a função train_model(),
    e retorna o modelo treinado.
    """
    X, y = load_dataset()
    model = train_model(X, y)
    print(model)
    
    return 


if __name__ == "__main__":
    main()
