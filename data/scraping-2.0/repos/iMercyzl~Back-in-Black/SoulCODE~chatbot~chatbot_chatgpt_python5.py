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
mensagem_inicial = """
Você: Olá, como posso te ajudar?
Modelo: Olá, eu sou um chatbot que pode te ajudar a encontrar o que você precisa.
"""

instrucao = "responder apenas sobre os recursos do chatbot"
mensagem_usuario = "quais sao os recursos do chatbot?"

conversa_com_instrucao = mensagem_inicial + "Você: " + mensagem_usuario + "\nInstrução: " + instrucao

resposta = enviar_mensagem(conversa_com_instrucao)