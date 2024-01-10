import openai

openai.api_key = ''

def create_message(role, content):
    return {'role': role, 'content': content}

def chat(agent1_message, agent2_message):
    messages = [
        create_message('system', 'You are chatting with two AI agents. They will take turns asking each other questions.'),
        create_message('user', agent1_message),
        create_message('assistant', agent2_message),
    ]

    while True:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        assistant_response = response['choices'][0]['message']['content']
        print(f'Agent 1: {messages[-1]["content"]}')
        print(f'Agent 2: {assistant_response}')
        messages.append(create_message('user', assistant_response))

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        assistant_response = response['choices'][0]['message']['content']
        print(f'Agent 1: {assistant_response}')
        print(f'Agent 2: {messages[-1]["content"]}')
        messages.append(create_message('assistant', assistant_response))

if __name__ == '__main__':
    chat('Hello, how are you?', 'I\'m fine, how about you?')
