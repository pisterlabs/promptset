import openai
import subprocess
from dotenv import load_dotenv
import os

load_dotenv()

# Pegando a chave da API do arquivo .env
openai.api_key = os.getenv("OPENAI_API_KEY")

def gerar_comando_shell(texto):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Escreva um comando shell que faça o seguinte: {texto}",
        temperature=0.7,
        max_tokens=2048,
        n=1,
        stop=None
    )
    return response['choices'][0]['text'].strip()

#Funcao que executa o comando no power shell
def executar_comando_shell(comando):
    try:
        resultado = subprocess.run(comando, shell=True, check=True)
        print(resultado)
    except subprocess.CalledProcessError as e:
        print(e)

descricao_comando = input("Digite uma descrição para o comando shell: ")
comando = gerar_comando_shell(descricao_comando)
print(f"Comando gerado: {comando}")

#Executando comando (perigoso)
#executar_comando_shell(comando)