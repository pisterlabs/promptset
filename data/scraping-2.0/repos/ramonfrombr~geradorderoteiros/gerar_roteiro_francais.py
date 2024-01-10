import openai
import time

openai.api_key = "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"

def criar_prompt_simples(sujet, langue):
    return f"Écrivez un script très court pour une vidéo qui enseigne {sujet} en {langue}. Le script doit être en français. Le scénario doit contenir ce que le présentateur dira mot pour mo"

def criar_prompt_detalhado(sujet, langue):
    return f"Écrivez un script détaillé pour une vidéo qui enseigne {sujet} en {langue}. Le script doit être en français. Le scénario doit contenir ce que le présentateur dira mot pour mot."

scripts1 = [
    "dire bonjour, bonjour, bon après-midi, bonne nuit",
    "dire au revoir, à demain, à plus tard, à la prochaine fois",
    "demande et réponds, quel est ton nom ?",
    "demandez et répondez d'où venez-vous?",
    "demandez et répondez où habitez-vous?",
    "Demandez et répondez quel âge avez-vous?"
    "Demandez et répondez au numéro de téléphone et à l'e-mail"
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
                        {"role": "system", "content": f"Vous êtes un assistant créant des scripts pour des vidéos d'enseignement des langues."},
                        {"role": "user", "content": criar_prompt_simples(roteiro, idioma)}
                    ],
                )

                generated_file = open(f'apprenez/{idioma}_{roteiro.replace(" ", "_").replace(",", "_").replace("?", "")}.txt', 'w')
                generated_file.writelines(response.choices[0].message.content)
                time.sleep(20)

            except openai.error.InvalidRequestError :
                print(f"there was an error when translating '{roteiro}': InvalidRequestError")
            
            except openai.error.APIError:
                print(f"there was an error when translating '{roteiro}': APIError")
            
            except openai.error.ServiceUnavailableError:
                print(f"there was an error when translating '{roteiro}': ServiceUnavailableError")

            time.sleep(20)

criar_roteiros("anglais", [scripts1])
criar_roteiros("espagnol", [scripts1])
criar_roteiros("alemán", [scripts1])
criar_roteiros("italien", [scripts1])
criar_roteiros("portugais", [scripts1])
