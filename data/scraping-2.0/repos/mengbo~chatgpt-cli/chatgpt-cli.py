import os
import openai

openai.api_base = os.environ.get('OPENAI_API_BASE')
openai.api_key = os.environ.get('OPENAI_API_KEY')
completion = openai.Completion()

start_chat_log = '''ME: 您好 
AI: 你好！很高兴能为你提供帮助。请问你有什么需要我帮忙的？
'''

def ask(question, chat_log=None):
    if chat_log is None:
        chat_log = start_chat_log
    prompt = f'{chat_log}ME: {question}\nAI:'
    response = completion.create(
        prompt=prompt, model="text-davinci-003",
        stop=['\nME'], temperature=0.0,
        top_p=1, frequency_penalty=0,
        presence_penalty=0.6, best_of=1,
        max_tokens=2048)
    answer = response.choices[0].text.strip()
    return answer

def append_chat_log(question, answer, chat_log=None):
    if chat_log is None:
        chat_log = start_chat_log
    return f'{chat_log}ME: {question}\nAI: {answer}\n'

def main():
    chat_log=None
    while '!quit' != (question := input("Me: ")):
        answer = ask(question, chat_log)
        print("AI: " + answer);
        chat_log = append_chat_log(
            question, answer, chat_log)

if __name__ == "__main__":
    main()
