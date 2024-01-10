import openai

def obter_resposta(pergunta):
    openai.api_key = ''

    resposta = openai.Completion.create(
        engine='text-davinci-003',
        prompt=pergunta,
        max_tokens=300,
        temperature=0.5,
        n=1,
        stop=None
    )

    return resposta.choices[0].text.strip()
