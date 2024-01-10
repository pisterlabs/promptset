import openai
import json


def parse_fullemail_with_chatbot(email_content_str: str, api_key: str) -> str:
    """
    Parses an email object string using OpenAI's GPT-3.5 Chat model to extract relevantemail_content_str: Your email content in a string
        api_key (str): Your OpenAI API key.
    Returns:
       a string that contains the formated content
    """
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": email_content_str +
                "\n--- maintain the original structure of the email and format the content for better readability"}
        ]
    )
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']
