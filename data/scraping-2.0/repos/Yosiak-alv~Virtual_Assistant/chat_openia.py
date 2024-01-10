import os
import openai

# Load your API key from an environment variable or secret management service
openai.api_key = ""

chat_history = []

def main():
    while True:
        prompt = input('Escribe un a pregunta: ')
        
        if(prompt == 'exit'):
            break
        else:
            chat_history.append({"role": "user", "content": prompt})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", 
                messages=chat_history,
                stream=True,
                max_tokens=100,
            )

            collection_messages = []

            for chunk in response:
                collection_messages.append(chunk['choices'][0]['delta'])
                full_message = ''.join([m.get('content','') for m in  collection_messages])
                print(full_message)
                print('\033[H\033[J',end='')  # Clear the screen

            chat_history.append({'role': 'assistant', 'content': full_message})
            print(full_message)

if __name__ == '__main__':
    main()