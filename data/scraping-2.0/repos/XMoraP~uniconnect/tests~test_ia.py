from flask.testing import FlaskClient
import pytest
import sys
import os
from unittest.mock import patch

# Obtener la ruta del directorio raíz del proyecto
current_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from app import app

# Fixture para el cliente de pruebas
@pytest.fixture
def client():
    app.config['TESTING'] = True
    client = app.test_client()
    yield client

# Test de la función 'get_openai_response' mockeada
@patch('app.get_openai_response')
def test_chat(mock_openai_response, client: FlaskClient):
    mock_openai_response.return_value = "Mocked response from OpenAI"

    data = {'msg': 'Hola, estoy probando'}
    response = client.post('/get', data=data)

    mock_openai_response.assert_called_once()
    assert response.status_code == 200
    assert b'Mocked response from OpenAI' in response.data

# Test para verificar el mensaje de bienvenida
def test_welcome_message(client):
    data = {'name': 'John'}
    response = client.post('/get_welcome_message', data=data)

    assert response.status_code == 200
    assert 'Bienvenido,' in response.data.decode('utf-8')
    assert '¿En qué puedo ayudarte?' in response.data.decode('utf-8')

# Test para verificar un mensaje vacío
def test_empty_message(client):
    data = {'msg': ''}
    response = client.post('/get', data=data)
    assert response.status_code == 400

# Nuevo test para el endpoint '/ask'
@patch('app.openai.Completion.create')
def test_ask_assistant(mock_openai_create, client: FlaskClient):
    mock_openai_create.return_value = {'choices': [{'text': 'Mocked response from OpenAI'}]}

    data = {'input': 'Hola, estoy probando'}
    response = client.post('/ask', json=data)

    mock_openai_create.assert_called_once_with(
        engine="text-davinci-003",
        prompt='Hola, estoy probando',
        max_tokens=100
    )
    assert response.status_code == 200
    assert response.json['response'] == 'Mocked response from OpenAI'
