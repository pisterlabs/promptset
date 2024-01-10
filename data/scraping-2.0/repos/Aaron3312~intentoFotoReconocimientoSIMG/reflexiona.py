import openai

# Configura tu clave de API
openai.api_key = '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'

# Establecer el primer prompt inicial
i = 0
# Bucle de diálogo iterativo
while True:
    # Solicitar al usuario que ingrese texto

    if i >= 1:
        usuario_entrada = input(f"reflexiona sobre esto {response.choices[0].text}")
    else:
        print ("Escribe algo para comenzar la reflexión:")
        usuario_entrada = input()
    # Actualizar el prompt con la entrada del usuario
    prompt = f"reflexiona sobre {usuario_entrada}\n"
    i += 1
    # Generar respuesta utilizando GPT-3
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        temperature=0.5,
        max_tokens=1000
    )
    

