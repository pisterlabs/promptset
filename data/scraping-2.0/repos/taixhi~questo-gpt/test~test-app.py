import pytest
import openai
from unittest.mock import patch
from server.main import app
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_generate_question_success(client):
    response = client.post('/generate_question', json={'prompt': 'Eiffel Tower'})
    print(response.get_json())
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'question' in json_data


def test_generate_question_no_prompt(client):
    response = client.post('/generate_question', json={})
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert json_data['error'] == 'Please provide a prompt.'

@patch("openai.ChatCompletion.create")
def test_generate_question_openai_failure(mock_create, client):
    # Let's simulate an error from OpenAI's API
    mock_create.side_effect = Exception("API Error")

    response = client.post('/generate_question', json={'prompt': 'Eiffel Tower'})
    # Depending on how your app handles exceptions, adjust the status_code and error message
    assert response.status_code == 500
