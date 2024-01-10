
import tkinter as tk
import openai
from langchain.llms import OpenAI

# Função para obter a chave de API da OpenAI a partir do arquivo
def obter_chave_api():
    try:
        with open('api.txt', 'r') as arquivo:
            chave_api = arquivo.read().strip()
        return chave_api
    except FileNotFoundError:
        raise Exception("O arquivo api_key.txt não foi encontrado. Certifique-se de criar o arquivo e adicionar sua chave de API.")

# Configurar a Langchain com a chave de API da OpenAI
openai.api_key = obter_chave_api()
llm = OpenAI()

# Função para carregar o contexto de um arquivo externo
def carregar_contexto():
    try:
        with open('contexto.txt', 'r') as arquivo:
            return arquivo.read().strip()
    except FileNotFoundError:
        return ""

contexto = carregar_contexto()

# Função para obter a resposta do chatbot
def chatbot_responder(pergunta):
    prompt = contexto + "\n\n" + pergunta if contexto else pergunta
    resposta = llm.complete(prompt=prompt, max_tokens=50)
    return resposta

# Função para responder à pergunta do usuário
def responder_pergunta():
    pergunta = entrada.get()
    resposta = chatbot_responder(pergunta)
    resposta_label.config(text=resposta)

# Configurar a janela
janela = tk.Tk()
janela.title("Chatbot Inteligente")

# Criar widgets
entrada = tk.Entry(janela, width=50)
botao = tk.Button(janela, text="Enviar Pergunta", command=responder_pergunta)
resposta_label = tk.Label(janela, text="Resposta aparecerá aqui.", wraplength=300)

# Colocar widgets na janela
entrada.pack()
botao.pack()
resposta_label.pack()

# Iniciar a interface gráfica
janela.mainloop()
