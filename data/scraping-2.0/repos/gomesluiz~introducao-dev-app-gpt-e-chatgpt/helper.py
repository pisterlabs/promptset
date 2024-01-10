import openai

def get_completion_from_messages(key,
    mensagens, 
    modelo="gpt-3.5-turbo", 
    temperature=0, 
    max_tokens=100
):
    openai.api_key = key    
    resposta = openai.chat.completions.create(
        model=modelo, 
        messages=mensagens, 
        temperature=temperature, 
        max_tokens=max_tokens
    )

    return resposta.choices[0].message.content