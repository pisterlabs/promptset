import openai
import os
import logging

class OpenAIClient:
    def __init__(self, api_key=None):
        if api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key is None:
                raise ValueError("API key not provided. Please set OPENAI_API_KEY environment variable or pass the key as argument.")
        else:
            self.api_key = api_key

        openai.api_key = self.api_key

    def prompt(self, query, model="text-davinci-002", max_tokens=50, temperature=0.5):
        try:
            response = openai.Completion.create(
                engine=model,
                prompt=query,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text
        except openai.error.AuthenticationError as e:
            logging.error(f"Error authenticating with OpenAI API: {e}")
        except openai.error.APIError as e:
            logging.error(f"Error communicating with OpenAI API: {e}")
        except Exception as e:
            logging.error(f"Unknown error occurred: {e}")

def test_openai_client():
    client = OpenAIClient()

    # Test prompt functionality
    response = client.prompt("What is the meaning of life?")
    assert "42" in response

    # Test error handling
    try:
        client = OpenAIClient(api_key="invalid_key")
    except ValueError:
        pass

    response = client.prompt("invalid_query")
    assert response is None

if __name__ == "__main__":
    test_openai_client()