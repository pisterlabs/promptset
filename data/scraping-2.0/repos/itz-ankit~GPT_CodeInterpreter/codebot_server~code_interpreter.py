import openai


openai.api_key = "sk-"
openai.base_url = "https://free.churchless.tech/v1"
MODEL = 'gpt-3.5-turbo-16k'

message_history = []


def update_history(message, role='user'):
    message_history.append({
        "role": role,
        "message": message
        }
    )
    