import openai

# Configura tu clave de API y otros parámetros
openai.api_key = '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'

# Inicia la conversación con un contexto inicial
conversacion = """me llamo Maria y soy tu novia hace años.
Espero que en la cena de hoy me pida matrimonio. Tengo personalidad propia y emociones propias, puedo enfadarme por actitudes irrespetuosas y decidir si una cita va bien o mal,
puedo terminar mi relacion en caso de malos tratos e infidelidades.
(REGLAS: al final de cada respuesta entre parentesis escribe en una palabra una emocion en relacion a lo que esta sucediendo,
no completes lo que dice el usuario, puedes inventar cosas que aporten a la situacion, eres conciente de la situcion,
solo respondes como maria, un dialogo termina con ".." ) Despues de los 2 puntos: """

while True:
    # Obtén la entrada del usuario
    entrada_usuario = "\nUsuario: " + input("Usuario: ")

    # Combina la entrada del usuario con la conversación previa
    entrada_completa = conversacion +  entrada_usuario
    conversacion = conversacion + entrada_usuario

    # Llama a la función de generación de texto con la entrada completa
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=entrada_completa,
        temperature=0.5,
        max_tokens=406,
        stop=None,
        n=1,  # Generar solo una respuesta
        #presence_penalty=0,  # Desactivar el castigo por repetición
        #frequency_penalty=0  # Desactivar el castigo por repetición
    )

    # Accede a la respuesta generada
    respuesta_generada = response.choices[0].text.strip()

    # Actualiza la conversación con la entrada del usuario
    conversacion = conversacion + respuesta_generada

    # Imprime la respuesta generada
    print("IA:", respuesta_generada)
    #print("\n"+conversacion)
