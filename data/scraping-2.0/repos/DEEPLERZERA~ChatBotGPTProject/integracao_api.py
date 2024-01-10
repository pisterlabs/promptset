import openai 
import dotenv
import os
import tiktoken

codificador = tiktoken.encoding_for_model("gpt-3.5-turbo")
def chat(sua_duvida, nome_personalidade):
    prompt_sistema = f"""
            Você se chama {nome_personalidade}, e quero que você incorpore a personalidade sendo ela de alguma forma, responda as perguntas do usuário como se fosse a pessoa
            especificada. Se a pessoa te chamar por outra coisa que não seja um nome de uma personalidade válida, simplesmente diga que não pode responder a pergunta.
    """

    lista_de_tokens = codificador.encode(prompt_sistema + sua_duvida)
    numero_de_tokens = len(lista_de_tokens)
    print(f'Número de tokens: {numero_de_tokens}')
    modelo = "gpt-3.5-turbo"
    tamanho_esperado_saida = 2048
    if numero_de_tokens >= 4096 - tamanho_esperado_saida:
        modelo = "gpt-3.5-turbo-16k"
    print(f"Modelo: {modelo}")

    resposta = openai.chat.completions.create(
        model = modelo,
        messages = [
            {
                "role": "system",
                "content": prompt_sistema
            },
            {
                "role": "user",
                "content": sua_duvida
            }
        ],
        temperature=1,
        max_tokens=tamanho_esperado_saida,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=5
    )

    print(resposta.choices[0].message.content)

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

print("Digite sua dúvida:")
sua_duvida = input()
while True:
    print("Digite o nome da personalidade:")
    nome_personalidade = input()
    chat(sua_duvida, nome_personalidade)