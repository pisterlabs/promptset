import gradio as gr
from langchain.llms import Ollama
import requests
from bs4 import BeautifulSoup

conversation_history = ["Answer prompts like you are a safety expert for industrial environments."]

def obter_texto_do_link(link):
    try:
        response = requests.get(link)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        texto = soup.get_text(separator='\n', strip=True)
        return texto
    except requests.exceptions.RequestException as e:
        return f"Erro ao obter conteúdo do link: {e}"

def generate_response(prompt, link_context):
    contexto = obter_texto_do_link(link_context)
    if "Erro" in contexto:
        return contexto
    
    print(contexto)

    conversation_history.append("Consider the following text as context: "+ contexto)
    conversation_history.append("Question: " + prompt)
    full_prompt = "\n".join(conversation_history)
    
    opa = Ollama(base_url='http://localhost:11434', model="orca-mini")
    resposta = opa(full_prompt)
    
    return resposta

def main():
    iface = gr.Interface(
        fn=generate_response,
        inputs=[gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
                gr.Textbox(placeholder="Enter the link for context")],
        outputs="text"
    )
    iface.launch()
    
if __name__ == "__main__":
    main()
