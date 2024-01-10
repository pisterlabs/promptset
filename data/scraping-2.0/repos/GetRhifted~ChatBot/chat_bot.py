import openai

openai.api_key = ''

while True:
    # Se define el motor de GPT-3 como 'text-davinci-003'
    engine_model_gpt4 = 'text-davinci-003'

    # Se solicita al usuario que ingrese un nuevo prompt
    prompt = input('Dime tu inquietud: ')

    # Si el prompt ingresado coincide con alguna de las palabras clave, se sale del bucle
    if prompt in ['exit', 'salir', 'quit', 'terminar']:
        break

    # Se crea una solicitud de completado de texto usando el modelo GPT-3
    completion = openai.Completion.create(
        engine = engine_model_gpt4,
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        stop = None,
        temperature = 0.3
    )

    # Se extrae el texto de la respuesta generada por GPT-3
    response = completion.choices[0].text

    # Se imprime la respuesta generada por GPT-3 en la consola
    print(response)
