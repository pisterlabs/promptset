import os
from openai import OpenAI
from os import environ

# Get the list of user's 
# environment variables 
env_var = os.environ["OPENAI_API_KEY"]

# Initialize the conversation list
conversation = [
    {"role": "user", "content": "Olá, como vai você?"},
]

# Inicialização do cli ente OpenAI
client = OpenAI(api_key=env_var)

def get_openai_response(question):
    model_engine = "gpt-4" # Escolha o modelo desejado

    # Adicione a nova mensagem para lista de conversas ao contexto
    conversation.append({"role": "user", "content": question})

    """Obtém a resposta do modelo da OpenAI."""
    response = client.chat.completions.create(
        model=model_engine,  # Escolha o modelo desejado
        messages=conversation,


    )

    # add the model`s response to the conversation list
    conversation.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content

def main():
    """Loop principal do chatbot."""
    print("Chatbot OpenAI (digite 'sair' para encerrar)")
    while True:
        user_input = input("Você: ")
        if user_input.lower() == 'sair':
            break
        response = get_openai_response(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
