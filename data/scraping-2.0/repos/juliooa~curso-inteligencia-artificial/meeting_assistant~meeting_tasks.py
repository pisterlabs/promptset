from openai_api import send_message

def summarize_meeting(transcription):
    system_message = '''
Tu eres un asistente experto en resumir reuniones. 
El usuario te va a enviar la transcripcion de una reunión y tu la vas a resumir.
Considera que en la reunión pueden haber varios interlocutores.
    '''
    return send_message(transcription,system_message)
    
def translate_transcription(transcription):
    system_message = '''
Tu eres un traductor Ingles-Español experto. Tu especialidad es traducir
transcripciones de reuniones, grabadas en inglés. 
El usuario te va a enviar la transcripcion de una reunión en ingles y tu
la vas a traducir al español.
Considera que en la reunión pueden haber varios interlocutores.
'''
    return send_message(transcription, system_message)

def get_actionable_items(meeting_transcription):
    system_message = '''
Tu eres un experto asistente en identificar items accionables que salgan de una reunión.
El usuario te va a enviar una transcripción de una reunión y tu vas a identificar
todas las acciones que hay que realizar una vez que la reunión termine.
Si es posible, vas a identificar por nombre quien tiene que hacer esas acciones,
si no es posible identificarlo, vas a poner "alguien" en su lugar.
La lista de acciones tienes que entregarla en español.
'''
    return send_message(meeting_transcription, system_message)