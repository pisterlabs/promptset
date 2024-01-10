from config.settings import OPEN_AI_KEY
from src.app import celery_app
import openai

openai.api_key = OPEN_AI_KEY
# openai.api_key = "your-openai-api-key-here" # Testing

completion = openai.Completion()

start_sequence = "The name is PocketGod\n\nPocketGod:"
restart_sequence = "\n\nPerson:"


def ask(question, chat_log=None):
    prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt_text,
        temperature=0.8,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
    )
    story = response['choices'][0]['text']
    return str(story)


def append_interaction_to_chat_log(question, answer, chat_log=None):
    return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'


@celery_app.task(name="get_answer")
def get_answer(question, chat_log):
    answer = ask(question, chat_log)
    return answer
