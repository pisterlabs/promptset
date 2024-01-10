import pytest
import openai
from unittest.mock import patch, Mock
from server.main import app
from server.helpers import write_log

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Test the /generate_question endpoint
# To use this single test, call `pytest test/test-app.py::test_generate_question_success`
data = """[{"question": "When was the Eiffel Tower completed?", "answer": "1889"}, {"question": "Who was the chief engineer responsible for the construction of the Eiffel Tower?", "answer": "Gustave Eiffel"}, {"question": "How tall is the Eiffel Tower?", "answer": "330 meters"}, {"question": "What was the purpose of building the Eiffel Tower?", "answer": "To serve as the entrance arch to the 1889 World's Fair"}, {"question": "How many steps are there to reach the top of the Eiffel Tower?", "answer": "1,665 steps"}, {"question": "What material was used to build the Eiffel Tower?", "answer": "Iron"}, {"question": "How long did it take to build the Eiffel Tower?", "answer": "2 years, 2 months, and 5 days"}, {"question": "What is the Eiffel Tower's original color?", "answer": "Red-brown"}, {"question": "How many visitors does the Eiffel Tower attract annually?", "answer": "Around 7 million"}, {"question": "What is the name of the restaurant located on the Eiffel Tower's top floor?", "answer": "Le Jules Verne"}]"""


@patch("openai.ChatCompletion.create")
def test_generate_question_success(mock_create, client):
    create_openai_object(data)
    mock_create.return_value = create_openai_object(data)
    
    response = client.post('/generate_question', json={'prompt': 'Eiffel Tower'})
    
    assert response.status_code == 200


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


# Snippet by @edwardselby on https://github.com/openai/openai-python/issues/398
def create_openai_object(payload):
   obj = openai.openai_object.OpenAIObject()
   message = openai.openai_object.OpenAIObject()
   content = openai.openai_object.OpenAIObject()
   content.content = payload
   message.message = content
   obj.choices = [message]
   return obj

