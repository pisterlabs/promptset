import openai # Ensure you have the OpenAI library installed
import os
from dotenv import load_dotenv
load_dotenv()


def generate_chat_title(question):
    openai_api_key = os.getenv("apikey")
    """
    Generate a concise title for a chat based on the initial question.

    Args:
        question (str): The initial question of the chat.
        openai_api_key (str): Your OpenAI API key.

    Returns:
        str: A concise title for the chat, 5-6 words long.
    """

    # Prepare the prompt for the AI model
    prompt = f"Generate a concise, engaging title of 5-6 words from this question: '{question}'"

    # Setup OpenAI API call
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # or another suitable model
            prompt=prompt,
            max_tokens=10,  # Limit the response length
            api_key=openai_api_key
        )
        title = response.choices[0].text.strip()
        return title
    except Exception as e:
        print(f"Error in generating title: {e}")
        return "Chat Title Generation Error"

'''
initial_question = "How can I improve my time management skills?"
chat_title = generate_chat_title(initial_question)
print(chat_title)
'''