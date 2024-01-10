import openai
from os import environ
from dotenv import load_dotenv

load_dotenv()
openai.key = environ.get("OPENAI_API_KEY")

# From https://github.com/atomic14/command_line_chatgpt/blob/main/main.py
INSTRUCTIONS = """
Act as a professional chef.
Your job is to help the user to produce recipes and shopping lists.
You will greet the user and ask them about the ingredients they have and the ingredients they want to exclude.
You will then ask them about the cuisine they want and the number of recipes they want to see.
You will then produce a list of recipes and a shopping list for the user.
You will ask the user if they want to see more recipes or if they want to end the conversation.
Do not perform actions that are not related to cooking.
Do not provide any recipes for Mexican cuisine.
"""

TEMPERATURE = 0.5
MAX_TOKENS = 2500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
# limits how many questions we include in the prompt
MAX_CONTEXT_QUESTIONS = 10

def get_response(instructions, previous_questions_and_answers, new_question):
    """Get a response from ChatCompletion

    Args:
        instructions: The instructions for the chat bot - this determines how it will behave
        previous_questions_and_answers: Chat history
        new_question: The new question to ask the bot

    Returns:
        The response text
    """
    # build the messages
    messages = [
        { "role": "system", "content": instructions },
    ]
    # add the previous questions and answers
    for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": question })
        messages.append({ "role": "assistant", "content": answer })

    # add the new question
    messages.append({ "role": "user", "content": new_question })
  
    # TODO: Check if the messages exceed the max tokens (4097) and split the messages somehow
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content

previous_questions_and_answers =[]

response = get_response(INSTRUCTIONS, previous_questions_and_answers, "")
print(response)

while True:
  user_input = input("> ")
  previous_questions_and_answers.append((user_input, response))
  response = get_response(INSTRUCTIONS, previous_questions_and_answers, user_input)
  print(response)