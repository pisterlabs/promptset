import openai
import time

openai.api_key = "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"

roteiros1 = [
    "dizer oi, bom dia, boa tarde, boa noite",
    "dizer tchau, até amanhã, até logo, até a próxima",
    "perguntar e responder qual é seu nome?",
    "perguntar e responder de onde você é?",
    "perguntar e responder onde você mora?",
    "perguntar e responder quantos anos você tem?",
    "perguntar e responder número de telefone e email"
]

roteiros2 = [
    "perguntar e responder como você está?",
    "pedir desculpas",
    "pedir licença",
    "falar números de 1 a 10",
    "falar números de 11 a 100",
    "perguntar e responder as horas",
    "perguntar e responder quanto custa?",
    "dizer números grandes",
    "falar dias, meses, estações, e datas"
]

roteiros3 = [
    "perguntar quem, o que, onde, quando, e por que",
    "fazer um pedido no restaurante",
    "falar sobre o clima",
    "falar sobre esportes",
    "pedir direções",
    "pedir um taxi",
    "pedir um café",
    "perguntar você fala português?",
    "pedir uma bebida"
]

roteiros4 = [
    "dizer feliz natal",
    "dizer que maneiro",
    "elogiar alguém",
    "dizer estados emocionais",
    "chamar alguém de idiota",
    "dizer não estou me sentindo bem",
    "dizer estou com fome e estou com sede",
    "dizer sim, não, claro, com certeza, nem pensar",
    "dizer eu gosto de você",
    "perguntar o que você está fazendo"
]


def criar_prompt_simples(topico, idioma):
    return f"Escreva um roteiro super pequeno para um vídeo que ensina {topico} em {idioma}. O roteiro deve estar em português. O roteiro deve conter o que o apresentador vai falar palavra por palavra."

def criar_prompt_detalhado(topico, idioma):
    return f"Escreva um roteiro detalhado para um vídeo que ensina {topico} em {idioma}. O roteiro deve estar em português. O roteiro deve conter o que o apresentador vai falar palavra por palavra."

def criar_roteiros(idioma, listas_de_roteiros):

    for lista in listas_de_roteiros:
        for roteiro in lista:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"Você é um assistente que criar roteiros para vídeos ensinando idiomas."},
                        {"role": "user", "content": criar_prompt_simples(roteiro, idioma)}
                    ],
                )

                generated_file = open(f'aprendaagora/{idioma}_{roteiro.replace(" ", "_").replace(",", "_").replace("?", "")}.txt', 'w')
                generated_file.writelines(response.choices[0].message.content)

            except openai.error.InvalidRequestError:
                print(f"there was an error when translating '{roteiro}'")

            except openai.error.APIError:
                print(f"there was an error when translating '{roteiro}': APIError")
            
            except openai.error.ServiceUnavailableError:
                print(f"there was an error when translating '{roteiro}': ServiceUnavailableError")

            time.sleep(20)

criar_roteiros("ingles", [roteiros2])

"""
criar_roteiros("espanhol", [roteiros1, roteiros2, roteiros3, roteiros4])
criar_roteiros("italiano", [roteiros1, roteiros2, roteiros3, roteiros4])
criar_roteiros("alemao", [roteiros1, roteiros2, roteiros3, roteiros4])
criar_roteiros("japones", [roteiros1, roteiros2, roteiros3, roteiros4])
criar_roteiros("chines", [roteiros1, roteiros2, roteiros3, roteiros4])
"""

"""
5 frases essenciais
hoje, ontem, amanhã
vocabulário de sala de aula
vocabulário de casa
vocabulário de cozinha
vocabulário de rotina
períodos do dia
guia de gramática 1
guia de gramática 2
conjugação
preposições
"""
