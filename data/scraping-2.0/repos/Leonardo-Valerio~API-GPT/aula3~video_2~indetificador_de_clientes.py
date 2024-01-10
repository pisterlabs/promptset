import os
import openai
import dotenv
import tiktoken
def carrega(nome_do_arquivo):
    try:
        with open(nome_do_arquivo, "r") as arquivo:
            dados = arquivo.read()
            return dados
    except IOError as e:
        print(f"Erro: {e}")

dotenv.load_dotenv()
openai.api_key = os.getenv("API_GPT")

prompt_sistema = """
Identifique o perfil de compra para cada cliente a seguir.

O formato de saída deve ser:

cliente - descreva o perfil do cliente em 3 palavras
"""

prompt_usuario = carrega("../dados/clientes_10.csv")
codificador = tiktoken.encoding_for_model("gpt-3.5-turbo")
lista_de_tokens = codificador.encode(prompt_usuario + prompt_sistema)
tokens = len(lista_de_tokens)
print(tokens)
modelo = "gpt-3.5-turbo"
tamanho_esperado_saida = 2048
if tokens > 4096 - tamanho_esperado_saida:
    modelo = "gpt-3.5-turbo-16k"
print(modelo)
resposta = openai.ChatCompletion.create(
  model=modelo,
  messages=[
    {
      "role": "system",
      "content": prompt_sistema
    },
    {
      "role": "user",
      "content": prompt_usuario
    }
  ],
  temperature=1,
  max_tokens=tamanho_esperado_saida,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(resposta.choices[0].message.content)
