import openai
from config import openai_apikey

openai.api_key = openai_apikey

def basic_chat(text):
    # Make a request to OpenAI for a response
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text,
        temperature=0.2,
        max_tokens=150,  # Adjust the number of tokens as needed
    )

    answer = response.choices[0].text
    return answer
