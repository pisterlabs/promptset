import openai
from app.services.openai_service import get_openai_api_key

def test_openai_authentication():
    # Replace 'your_openai_api_key' with the actual OpenAI API key
    openai_api_key = get_openai_api_key()
    openai.api_key = openai_api_key

    # Perform a test API request to check if authentication works
    response = openai.Completion.create(
        engine="davinci",
        prompt="This is a test prompt.",
        max_tokens=5
    )

    assert response['object'] == 'text_completion'
