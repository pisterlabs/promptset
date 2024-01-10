from openai import OpenAI
import tiktoken
client = OpenAI()

encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

def load(file_name):
    try:
        with open(file_name, "r") as file:
            data = file.read()
            return data
    except IOError as e:
        print(f"Error: {e}")

def identifyProfile():
    system_prompt = f"""
    Identifique o perfil de compra para cada cliente a seguir.

    O formato de saída deve ser:

    cliente - descreva o perfil do cliente em até 3 palavras
    """

    user_prompt = load("./gpt-python-1-dados/lista_de_compras_10_clientes.csv")

    tokens_list = encoder.encode(system_prompt + user_prompt)
    tokens_number = len(tokens_list)
    print(f"Números de tokens na entrada: {tokens_number}")
    model = "gpt-3.5-turbo"
    output_size_expected = 2048
    if tokens_number >= 4096 - output_size_expected: # 4096 number from open ai site, token limit of gpt 3.5 turbo 
        model = "gpt-3.5-turbo-16k"

    print(f"Modelo escolhido: {model}")

    response = client.chat.completions.create(
    model=model,
    messages=[
        {
        "role": "system",
        "content": system_prompt
        },
        {
        "role": "user",
        "content": user_prompt
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    #   n=5
    )
    print(response.choices[0].message.content)

identifyProfile()