import openai
import pyttsx3

# Configurar la clave de la API de OpenAI
openai.api_key = "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"
def generar_respuesta(pregunta):
    prompt = "Pregunta: {}\nRespuesta:,  /// Eres un asistente llamado TAMARA y ayudas a discapacitados visuales"
    prompt_with_question = prompt.format(pregunta)
    respuesta = openai.Completion.create(
        engine="text-davinci-003",  # Selecciona el modelo de GPT-3 que deseas utilizar
        prompt=prompt_with_question,
        max_tokens=50,  # Define el límite de longitud de la respuesta generada
        n=1,  # Número de respuestas generadas
        stop=None,  # Criterio opcional para detener la generación de texto
        temperature=0.7,  # Controla la aleatoriedad de la respuesta generada
    )
    return respuesta.choices[0].text.strip()

def main():
    while True:
        # Esperar la entrada del usuario
        pregunta = input("Hazme una pregunta (o escribe 'salir' para terminar): ")

        if pregunta.lower() == "salir":
            break

        # Generar respuesta con GPT-3
        respuesta_generada = generar_respuesta(pregunta)

        # Imprimir la respuesta generada por GPT-3
        print("Respuesta generada por GPT-3:", respuesta_generada)

        # Pronunciar la respuesta generada utilizando pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('voice', 'spanish')  # Establecer el idioma a español
        engine.say(respuesta_generada)
        engine.runAndWait()

if __name__ == "__main__":
    main()