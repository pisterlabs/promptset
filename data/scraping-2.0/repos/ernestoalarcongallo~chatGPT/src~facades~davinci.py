import openai


def authenticate(key: str):
    """
    authenticate in openAI with your API key
    :param key: (string) the openAI provided secret API key
    :return: None
    """
    openai.api_key = key


def generate_response(question: str) -> str:
    """
    Send a prompt to the GPT model and get a response
    :param question: (string) user's formulated question
    :return: (string) the model's answer
    """
    # Set up the parameters for the API request
    completions = openai.Completion.create(
        engine="text-davinci-002",
        prompt=question,
        max_tokens=1024,
        temperature=0.5,
    )

    # Get the response from the API request
    message = completions.choices[0].text
    # Remove any leading or trailing white space from the response
    message = message.strip()

    return message
