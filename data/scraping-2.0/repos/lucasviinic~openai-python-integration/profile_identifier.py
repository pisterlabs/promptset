import os
import openai
import dotenv
import tiktoken

coder = tiktoken.encoding_for_model("gpt-3.5-turbo")

def load(file_name: str) -> str:
    try:
        with open(file_name, "r") as file:
            data = file.read()
            return data
    except IOError as e:
        print(f"Error: {e}")

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

prompt_system = """
Identifique o perfil de compra para cada cliente a seguir.

O formato de saída deve ser:

cliente - descreva o perfil do cliente em 3 palavras
"""

prompt_user= load("./data/lista_de_compras_100_clientes.csv")

token_list = coder.encode(prompt_system + prompt_user)
token_number = len(token_list)

print(f"Número de tokens na entrada: {token_number}")

expected_output_size = 2048
model = "gpt-3.5-turbo" 
if token_number < 4096 - expected_output_size:
    model = "gpt-3.5-turbo-16k"

print(f"Modelo escolhido: {model}")

resposta = openai.chat.completions.create(
  model=model,
  messages=[
    {
      "role": "system",
      "content": prompt_system
    },
    {
      "role": "user",
      "content": prompt_user
    }
  ],
  temperature=1,
  max_tokens=expected_output_size,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(resposta.choices[0].message.content)
