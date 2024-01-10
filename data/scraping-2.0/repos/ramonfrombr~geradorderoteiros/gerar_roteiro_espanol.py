import openai
import time

openai.api_key = "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"

def criar_prompt_simples(tema, idioma):
    return f"Escribe un guión súper corto para un video que enseñe {tema} en {idioma}. El guión debe estar en español. El guión debe contener lo que el presentador dirá palabra por palabra."

def criar_prompt_detalhado(tema, idioma):
    return f"Escribe un guión detallado para un vídeo que enseñe {tema} en {idioma}. El guión debe estar en español. El guión debe contener lo que el presentador dirá palabra por palabra."

scripts1 = [
    "decir hola, buenos días, buenas tardes, buenas noches",
    "decir adiós, hasta mañana, hasta luego, hasta la próxima",
    "pregunta y responde ¿cómo te llamas?",
    "pregunta y responde ¿de dónde eres?",
    "pregunta y responde ¿dónde vives?",
    "pregunta y responde ¿cuántos años tienes?",
    "Preguntar y contestar número de teléfono y correo electrónico"
]

scripts2 = [
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

scripts3 = [
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

scripts4 = [
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

def criar_roteiros(idioma, listas_de_roteiros):

    for lista in listas_de_roteiros:
        for roteiro in lista:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"Eres un asistente que crea guiones para vídeos de enseñanza de idiomas."},
                        {"role": "user", "content": criar_prompt_simples(roteiro, idioma)}
                    ],
                )

                generated_file = open(f'aprendaahora/{idioma}_{roteiro.replace(" ", "_").replace(",", "_").replace("?", "")}.txt', 'w')
                generated_file.writelines(response.choices[0].message.content)
                time.sleep(20)

            except openai.error.InvalidRequestError :
                print(f"there was an error when translating '{roteiro}': InvalidRequestError")
            
            except openai.error.APIError:
                print(f"there was an error when translating '{roteiro}': APIError")
            
            except openai.error.ServiceUnavailableError:
                print(f"there was an error when translating '{roteiro}': ServiceUnavailableError")

            time.sleep(20)

criar_roteiros("italiano", [scripts1])
criar_roteiros("portugués", [scripts1])
