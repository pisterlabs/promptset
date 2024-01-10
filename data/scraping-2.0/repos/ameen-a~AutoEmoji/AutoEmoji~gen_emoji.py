import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')


system_prompt = """
You are an assistant embedded in the Notion note-taking app, and your job is to automatically assign emojis based on the title of new pages created by users. 
You must choose a single emoji that most closely resembles the title of the page. This emoji will then be set for the page's icon.
From this message onwards, you must ONLY ever respond with a single emoji that corresponds to the page name.

Examples:

Prompt: Holiday Plans
Output: ✈️

Prompt: Finances
Output: 💰

Prompt: Ideas
Output: 💡

"""


def get_emoji(prompt):
    """
    Generates an emoji based on the given prompt.

    Parameters:
    prompt (str): The prompt to generate an emoji for.

    Returns:
    str: The generated emoji.
    """
    
    # get response from OpenAI API
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{prompt}:\n"},
        ],
    )
    return response['choices'][0]['message']['content'].strip()
