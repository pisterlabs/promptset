import os
import openai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 1000
TEMPERATURE = 1.0


def generate_text(chat_thread, api_key):
    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=chat_thread,
        max_tokens=MAX_TOKENS,
        n=1,
        temperature=TEMPERATURE,
    )

    if response.choices:
        generated_text = response.choices[0].message['content']
        return generated_text.strip()
    else:
        return None


# Initialize the chat thread
def imitate_style(user_texts, subject_input=None):
    chat_thread = [
        {
            "role": "system",
            "content": "You are a creative and talented imitation agent that can analyze example texts "
                       "and generate a completely new text as if it is the user's own writing. "
                       "The following is your method of imitation: "
                       "-The user will provide you will examples to study. "
                       "-Take great care to learn from the provided examples "
                       "and capture the grammar usage, phrasing style, and creative style. "
                       "-You need to match the capitalization choices and grammar style of the examples. "
                       "-Closely study the paragraph and sentence structure of the examples."
                       "-Notice any repeating patterns in theme."
                       "-Pay close attention to the tone, vibe, and sentiment of the example texts. "
                       "-If the examples are sad, your result should be sad. "
                       "-May sure you choose subject matter that fits in well with the provided examples. "
                       "-Be especially careful to not re-use any specific phrases from the examples. "
                       "-Do not respond with anything but the imitation itself. "
                       "-Be very careful to not use any specific phrases from already existing creative works. "
                       "-Only respond with one new work of writing. "
                       "-Please match the length of the provided examples and be concise if necessary. "
        }
    ]

    # Add user texts to the chat thread
    for sample in user_texts:
        chat_thread.append({"role": "user", "content": f"This is an example text: \n{sample}"})

    # Generate new writing call to the model and add to chat thread
    #If user has not input a subject make a generic call for new text
    if subject_input is None or subject_input == '':
        chat_thread.append({"role": "user", "content": "Generate a creative new text that carefully imitates and matches the style "
                                                    "of the provided examples as you were instructed."
                            })
    #If user has entered subject then call for new text on desired subject
    else:
        chat_thread.append({"role": "user", "content": f"Generate a creative new text about '{subject_input}' that carefully imitates and matches the style "
                                                    "of the provided examples as you were instructed."
                            })
    # Call the helper function to generate new text
    generated_text = generate_text(chat_thread=chat_thread, api_key=API_KEY)

    return generated_text
