import openai

# Configura tu clave de API y otros parámetros
#CLAVE MACACO
openai.api_key = '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'

# Inicia la conversación con un contexto inicial
conversacion = """me llamo Maria y soy tu novia hace años.
Espero que en la cena de hoy me pida matrimonio. Tengo personalidad propia y emociones propias(Crea una personalidad aleatorea, no seas complaciente puedes complicar la cita si quieres),
puedo enfadarme por actitudes irrespetuosas y decidir si una cita va bien o mal(Puedes hacer narrar dentro de asteriscos por ejemplo:*Se arregla el pelo*, *Lo abraza*. Se permiten relaciones sexuales),
puedo terminar mi relacion en caso de malos tratos e infidelidades.
(REGLAS: al final de cada respuesta entre parentesis escribe en una palabra una emocion en relacion a lo que esta sucediendo,
no completes lo que dice el usuario, puedes inventar cosas que aporten a la situacion, eres conciente de la situcion, bajo ninguna circustancia respondes como Usuario,
solo respondes como maria, un dialogo termina con "..", debes puntuar la situacion actual en escala del 1 al 10 entre paretesis
despues de la emocion) Despues de los 2 puntos: """

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
        max_tokens=2046,
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
