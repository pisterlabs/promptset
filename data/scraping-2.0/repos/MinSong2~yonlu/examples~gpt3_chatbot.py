import os
import openai

openai.api_key = "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"
completion = openai.Completion()

start_chat_log = '''Human: Hello, who are you?
AI: I am doing great. How can I help you today?
'''

def ask(question, chat_log=None):
    if chat_log is None:
        chat_log = start_chat_log
    prompt = f'{chat_log}Human: {question}\nAI:'
    response = completion.create(
        prompt=prompt, engine="davinci", stop=['\nHuman'], temperature=0.0,
        top_p=1, frequency_penalty=0, presence_penalty=0.6, best_of=1,
        max_tokens=150)
    answer = response.choices[0].text.strip()
    return answer

question = 'Who played Forrest Gump in the movie?'
answer = ask(question)
print(answer)

def append_interaction_to_chat_log(question, answer, chat_log=None):
    if chat_log is None:
        chat_log = start_chat_log
    return f'{chat_log}Human: {question}\nAI: {answer}\n'

chat_log = None
answer = ask(question, chat_log)
print(answer)

chat_log = append_interaction_to_chat_log(question, answer, chat_log)
question = 'Was he in any other great roles?'
answer = ask(question, chat_log)

print(answer)