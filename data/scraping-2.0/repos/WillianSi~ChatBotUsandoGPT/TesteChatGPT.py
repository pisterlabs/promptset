import openai

openai.api_key = "sua_chave_API"

def perguntar_responder(pergunta):
    prompt = f"Q: {pergunta}\nA:"
    
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None
    )
    
    resposta = response.choices[0].text.strip()
    return resposta

while True:
    pergunta = input("Faça sua pergunta (ou digite 'sair' para encerrar): ")
    
    if pergunta.lower() == "sair":
        break
    
    resposta = perguntar_responder(pergunta)
    print(resposta)