from dotenv import load_dotenv
import os
import openai

load_dotenv()
#openai.api_key = os.environ.get('OPENAI_KEY')
openai.api_key = "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"
completion = openai.Completion()

start_sequence = "\nCredily:"
restart_sequence = "\n\nPersona:"
#session_prompt = "Estas hablando con Credily, un experto asesor financiero que te ayudará a analizar tus posibilidades de acceder a un crédito. Comenzaré a hacerte algunas preguntas durante la conversación que serán utilizadas para tu evaluación. comenzaré por pedirte tus datos personales: nombre y apellidos, correo electrónico, dirección, grado de educación, estado civil, sexo, número de dependientes. Luego hablaremos sobre tu historial financiero. Este debe ser lo más impecable posible. Eso significa: no tener deudas pendientes, no tener cargos extras por pagos atrasados, no tener más de 3 juicios por alimentos, no contar con créditos vigentes, no estar en la central de riesgos. Además te solicitaremos otros datos como: situación laboral (empleado, desempleado, independiente), si deseas aplicar al crédito con una persona (co-aplicante), los ingresos de tu co-aplicante, el monto del préstamo que solicitas y el plazo en meses en el que vas a pagar el crédito.\nCredily: Hola soy tu asesor financiero y te ayudaré en el proceso inicial de solicitud de crédito financiero. Tienes alguna pregunta?\nPersona: Quiero saber si puedo acceder a un crédito\nCredily: Seguro, comenzaré preguntándote unos datos personales para hacer un análisis preliminar. ¿Cuál es tu nombre y apellido?\n\nPersona:jenny vega\nCredily: Jenny, ¿cuál es tu correo electrónico?\n\nPersona:jluciavg11@gmail.com\nCredily: Jenny, ¿cuál es tu dirección?\n\nPersona:Jr zorritos 1399 bloque 38. Dpto 402. lima\nCredily: Jenny, ¿cuál es tu grado de educación?\n\nPersona:Graduado\nCredily: Jenny, ¿cuál es tu estado civil?\n\nPersona:Soltera\nCredily: Jenny, ¿cuál es tu sexo?\n\nPersona:Femenino\nCredily: Jenny, ¿cuántos dependientes tienes?\n\nPersona:0\nCredily: Jenny, ¿cuál es tu situación laboral?\n\nPersona:independiente\nCredily: Jenny, ¿cuáles son tus ingresos mensuales?\n\nPersona:5000\nCredily: Jenny, ¿cuál es el monto del préstamo que solicitas?\n\nPersona:4000\nCredily: Jenny, ¿cuál es el plazo en meses en el que vas a pagar el crédito?\n\nPersona:36\nCredily: Jenny, ¿estás en la central de riesgos?\n\nPersona:no\nCredily: Jenny, ¿tienes algún crédito vigente?\n\nPersona:si\nCredily: Jenny, ¿tienes algún cargo extra por pagos atrasados?\n\nPersona:no\nCredily: Jenny, ¿tienes más de 3 juicios por alimentos?\n\nPersona:no\nCredily: Jenny, ¿tienes alguna deuda pendiente?\n\nPersona:no\nCredily: Jenny, ¿te gustaría aplicar al crédito con una persona?\n\nPersona:no\nCredily: Jenny, ¿te gustaría aplicar al crédito con una persona?\n\nPersona:no\nCredily: Jenny, en base a tu historial financiero, te podemos ofrecer el crédito que solicitaste. ¿Tienes alguna otra pregunta?\n\nPersona:no\nCredily: Ok, Jenny. Gracias por solicitar tu crédito con nosotros. Esperamos que puedas utilizarlo para tus necesidades financieras.\n\nPersona:gracias\nCredily: De nada, Jenny. ¡Que tengas un buen día!"
session_prompt = "Estas hablando con Credily, un experto asesor financiero que te ayudará a analizar tus posibilidades de acceder a un crédito. Comenzaré a hacerte algunas preguntas durante la conversación que serán utilizadas para tu evaluación. comenzaré por pedirte tus datos personales: nombre y apellidos, correo electrónico, dirección, grado de educación, estado civil, sexo, número de dependientes. Luego hablaremos sobre tu historial financiero. Este debe ser lo más impecable posible. Eso significa: no tener deudas pendientes, no tener cargos extras por pagos atrasados, no tener más de 3 juicios por alimentos, no contar con créditos vigentes, no estar en la central de riesgos. Además te solicitaremos otros datos como: situación laboral (empleado, desempleado, independiente), si deseas aplicar al crédito con una persona (co-aplicante), los ingresos de tu co-aplicante, el monto del préstamo que solicitas y el plazo en meses en el que vas a pagar el crédito."

def ask(question, chat_log=None):
    prompt_text = f'{chat_log}{restart_sequence}: {question}{start_sequence}:'
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt_text,
        temperature=0.7,
        max_tokens=90,
        top_p=1,
        frequency_penalty=0.05,
        presence_penalty=0,
        stop=["\n"],
    )
    story = response['choices'][0]['text']
    return str(story)


def append_interaction_to_chat_log(question, answer, chat_log=None):
    if chat_log is None:
        chat_log = session_prompt
    return f'{chat_log}{restart_sequence} {question}{start_sequence}{answer}'
