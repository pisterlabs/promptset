from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest

from src.services.exceptions import ApiKeyException
from src.summarization.engines import LangChain, TensorFlow


def test_langchain_summarize():
    with patch("langchain.llms.openai.OpenAI.predict") as mock_predict:
        # Configurar o retorno mockado
        mock_predict.return_value = "Resumo Mockado"
        langchain = LangChain(api_key="fake-api-key")

        # Texto de entrada
        input_text = "Texto de exemplo"
        result = langchain.summarize(input_text)
        assert result == "Resumo Mockado"


@patch("requests.post")
def test_token_is_valid_success(mock_post):
    # Configurar o mock para simular uma resposta bem-sucedida
    mock_post.return_value.status_code = 200
    langchain = LangChain(api_key="fake-api-key")

    try:
        langchain.token_is_valid()  # Não deve levantar exceção
    except ApiKeyException:
        pytest.fail("ApiKeyException não deveria ter sido levantada")


@patch("requests.post")
def test_token_is_valid_failure(mock_post):
    # Configurar o mock para simular uma resposta de falha (e.g., não autorizado)
    mock_post.return_value.status_code = HTTPStatus.UNAUTHORIZED
    langchain = LangChain(api_key="fake-api-key")

    with pytest.raises(ApiKeyException):
        langchain.token_is_valid()


@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.TFAutoModelForSeq2SeqLM.from_pretrained")
def test_tensorflow_summarize(mock_model_from_pretrained, mock_tokenizer_from_pretrained):
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    mock_tokenizer_from_pretrained.return_value = mock_tokenizer
    mock_model_from_pretrained.return_value = mock_model

    # Configurar o retorno mockado para o tokenizer e o modelo
    mock_tokenizer.encode.return_value = "Tensor Mockado"
    mock_model.generate.return_value = [42]  # ID fictício
    mock_tokenizer.decode.return_value = "Resumo Mockado"

    # Criar uma instância da classe TensorFlow e chamar o método summarize
    tensorflow = TensorFlow(model_name="fake-model")
    result = tensorflow.summarize("Texto de exemplo")

    # Verificar se os métodos do tokenizer e do modelo foram chamados corretamente
    mock_tokenizer.encode.assert_called_once()
    mock_model.generate.assert_called_once()
    mock_tokenizer.decode.assert_called_once_with(42, skip_special_tokens=True)

    # Verificar se o resultado é o esperado
    assert result == "Resumo Mockado"
