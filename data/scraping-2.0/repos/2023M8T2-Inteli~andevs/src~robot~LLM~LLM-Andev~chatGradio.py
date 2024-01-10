import gradio as gr
from langchain.llms import Ollama
import requests
import json
import re
from chatbot_actions import chatbot_print

intent_dict = {
    r"Indo pegar [ao]?\s*([\w\s]+)": chatbot_print
}

def send_request(question):
    url = "http://localhost:11434/api/generate"

    # Dados a serem enviados no corpo da solicitação
    data = {
        "model": "Alfred3",
        "prompt": question,
        "stream": False
    }
    # Faz a solicitação POST usando a biblioteca requests
    response = requests.post(url, json=data)

    # Verifica se a solicitação foi bem-sucedida (código de status 200)
    if response.status_code == 200:
        # Exibe a resposta do servidor
        response = vars(response)
        response['_content'] = json.loads(response["_content"].decode('utf-8'))

        return str(response["_content"]["response"])

    else:
        # Se a solicitação não for bem-sucedida, exibe uma mensagem de erro
        print(response.text)
        return f"Erro: {response.status_code}"

# Função para gerar resposta do modelo GPT-3
def generate_response(prompt, chat_history):
    print('Making request...')
    response = send_request(prompt)
    print(response)
    chat_history.append((prompt, response))
    for key, function in intent_dict.items():
        pattern = re.compile(key)
        point = pattern.findall(response)
        if point:
            print("Ação detectada")
            print(point)
            function(point[0])
    
    return "", chat_history

with gr.Blocks() as demo:
    title="LLM Chatbot"
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(generate_response, [msg, chatbot], [msg, chatbot])

# Execute a interface Gradio
if __name__ == "__main__":
    demo.launch()