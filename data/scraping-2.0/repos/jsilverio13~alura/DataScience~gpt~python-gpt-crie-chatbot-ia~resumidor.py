import dotenv
import os
import openai

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def resumidor_de_historico(historico):
    resposta_resumidor = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
            "role": "user",
            "content": f"""
            Resumir progressivamente as linhas de conversa fornecidas, 
            acrescentando ao resumo anterior e retornando um novo resumo. 
            Não apague nenhum assunto da conversa. 
            Se não houver resumo, apenas continue a conversa normalmente.

            ## EXEMPLO:
            O usuario pergunta o que a IA pensa sobre a inteligência artificial. 
            A IA acredita que a inteligência artificial é uma força para o bem.
            Usuário: Por que você acha que a inteligência artificial é uma força para o bem?
            IA: Porque a inteligência artificial ajudará os humanos a alcançarem seu pleno 
            potencial.

            ### Novo resumo:
            O usuario questiona a razão pela qual a IA considera a inteligência artificial 
            uma força para o bem, e a IA responde que é porque a inteligência artificial 
            ajudará os humanos a atingirem seu pleno potencial.

            ## FIM DO EXEMPLO
            
            Resumo atual:
            {historico}

            Novo resumo:"""
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return resposta_resumidor

def criando_resumo(historico):
    resposta = resumidor_de_historico(historico=historico)
    resumo = resposta.choices[0].message.content
    return resumo