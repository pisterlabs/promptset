import openai


openai.api_key = "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"


historial = [{"role": "user", "content": "Hola, necesito que actues como psicologo"}]


def mandar_mensaje(mensaje: str):
    global historial
    historial.append({"role": "user", "content": mensaje})
    respuesta = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=historial,
    )
    historial.append({"role": "assistant", "content": respuesta["choices"][0]["message"]["content"]})
    return respuesta["choices"][0]["message"]["content"]