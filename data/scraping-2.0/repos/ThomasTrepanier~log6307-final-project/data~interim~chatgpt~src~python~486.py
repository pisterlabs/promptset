import openai
from app.config.settings import settings

def test_openai_authentication():
    # Replace 'your_openai_api_key' with the actual OpenAI API key
    openai.api_key = settings.OPENAI_API_KEY

    # Perform a test API request to check if authentication works
    response = openai.Completion.create(
        engine="davinci", prompt="This is a test prompt.", max_tokens=5
    )

    assert response["object"] == "text_completion"

    # Test if authentication organization matches
    assert response["usage"]["organization"] == settings.OPENAI_ORGANIZATION

    # Print a success message if authentication is successful
    print("OpenAI authentication successful!")

    # Further tests and assertions related to OpenAI interactions can be added here
