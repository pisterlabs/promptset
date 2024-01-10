import openai
openai.api_key = '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'

def enviar_mensagem(mensagem):
    resposta = openai.Completion.create(
        engine="text-davinci-003",
        prompt=mensagem,
        max_tokens=50,
        temperature=0.7,
        n=1,
        stop=None
    )
    return resposta.choices[0].text.strip()
    
    mensagem_usuario = input('em que posso te ajudar?')

    resposta = enviar_mensagem(mensagem_usuario)

    if 'Desculpe, não entendi' in resposta:
        mensagem_usuario = "você pode fornecer mais informações?"
        resposta = enviar_mensagem(mensagem_usuario)
    print(f'resposta do chatgpt: {resposta}')