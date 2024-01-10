import openai

def get_initial_message():
    '''
    (Function)
        Esta funcion genera la lista de diccionarios con mensajes iniciales
        para poder pre-cargar de manera correcta la IA.
        Aqui se asigna el rol del sistema, del usuario y del asistente.
    '''
    messages=[
            {"role": "system", "content": "You are a helpful AI Tutor. Who anwers brief questions about AI."},
            {"role": "user", "content": "I want to learn AI"},
            {"role": "assistant", "content": "Thats awesome, what do you want to know aboout AI"}
        ]
    return messages

def get_chatgpt_response(messages, model="gpt-3.5-turbo"):
    '''
    (Function)
        Esta funcion genera la respuesta de un modelo en base al hilo de 
        la conversacion.
    (Parameters)
        - messages: [Clase de streamlit] la cual contiene el hilo de la conversacion
        - model: [str] El modelo que usaremos para que nos conteste
    (Returns)
        - [str] que es la respuesta del modelo al query ingresado.
    '''
    # Imprime el modelo que va a responder el query
    print("model: ", model)
    # Se genera la respuesta del modelo, en base al hilo de mensajes
    response = openai.ChatCompletion.create(
        # Modelo a usar
                        model=model,
        # Hilo de conversacion
                        messages=messages
                        )
    
    '''En este momento response es un json que se parece a lo siguiente
    {
    "choices": [
        {
        "finish_reason": "stop",
        "index": 0,
        "message": {
            "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
            "role": "assistant"
        }
        }
    ],
    "created": 1677664795,
    "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
    "model": "gpt-3.5-turbo-0613",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 17,
        "prompt_tokens": 57,
        "total_tokens": 74
    }
    }
    Por lo que lo que necesitamos (mensaje) se obtiene con el siguiente comando'''
    return  response['choices'][0]['message']['content']

def update_chat(messages, role, content):
    '''
    (Function)
        Esta funcion actualiza los mensajes de la conversacion bot-human
    (Parameters)
        - messages: [Clase de streamlit] la cual contiene el hilo de la conversacion
        - role: [str] Quien emite el mensaje
        - content: [str] Contenido del mensaje
    (Returns)
        - messages (vea parameters de esta funcion para mas info.)
    '''
    messages.append({"role": role, "content": content})
    return messages
