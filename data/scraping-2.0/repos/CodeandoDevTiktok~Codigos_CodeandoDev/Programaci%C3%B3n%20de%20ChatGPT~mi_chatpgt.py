import openai

#Configura Key API

openai.api_key = "pon-tu-api-key"

#Función de chat

def enviar_conversacion(mensajes):
    respuesta = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = mensajes
    )
    
    return respuesta.choices[0].message.content

#Funcion principal del chatbot

def chatbot():
    print("Bienvenido al chat, escribe tus mensajes")
    print("Escribe 'salir' para terminar la conversación")
    
    mensajes = []
    
    while True:
        mensaje = input("Usuario: ")
        
        if mensaje.lower() == "salir":
            break
        
        mensajes.append({"role":"user","content":mensaje})
        respuesta = enviar_conversacion(mensajes)
        
        mensajes.append({"role":"assistant","content":respuesta})
        print("Chatbot: ", respuesta)
    print("Chatbot: hasta luego")
    
chatbot()
