import pytest
import os
import train_model

@pytest.fixture(scope="module")
def train_and_save_model():
    """
    Fixture para treinar e salvar o modelo antes de executar o teste test_model_file_exists.

    Esta função chama a função main() do módulo train para treinar e salvar o modelo.
    Após o término do teste, remove o arquivo 'model.pkl'.
    """
    train_model.main()
    yield
    os.remove('model.pkl')

def test_model_file_exists(train_and_save_model):
    """
    Testa se o arquivo model.pkl foi salvo corretamente.

    Este teste, que depende da fixture train_and_save_model, verificará a existência do arquivo model.pkl no diretório atual.
    Se o arquivo não existir, o teste falhará com uma mensagem indicando que o arquivo model.pkl não foi salvo corretamente.
    """
    assert os.path.isfile('model.pkl'), "O arquivo model.pkl não foi salvo corretamente"
