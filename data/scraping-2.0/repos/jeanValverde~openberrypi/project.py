import os
import openai
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import cv2
import pyfiglet

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is not None:
    openai.api_key = OPENAI_API_KEY

    # Función para convertir el texto a voz
    def text_to_speech(text):
        speech = gTTS(text, lang="es")
        speech.save('response.mp3')
        playsound('response.mp3')

    # Función para reconocer el comando de voz
    def recognize_speech():
        recognizer = sr.Recognizer()

        with sr.Microphone() as source:
            print("Diga algo...")
            audio = recognizer.listen(source)

        try:
            print("Reconociendo...")
            text = recognizer.recognize_google(audio, language="es")
            print("Comando de voz:", text)
            return text
        except sr.UnknownValueError:
            print("No se pudo reconocer el comando de voz.")
        except sr.RequestError as e:
            print(f"Error al reconocer el comando de voz: {str(e)}")

    # Mostrar imagen de "cargando"
    loading_image = cv2.imread('loading.png')
    cv2.imshow("Cargando...", loading_image)
    cv2.waitKey(1)

    # Obtener el mensaje de entrada por comando de voz
    user_message = recognize_speech()

    # Enviar el mensaje a ChatGPT
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    # Obtener la respuesta de ChatGPT
    chat_response = completion.choices[0].message['content']
    print("Respuesta de ChatGPT:", chat_response)

    # Cerrar la ventana de "cargando"
    cv2.destroyAllWindows()

    # Convertir la respuesta a voz
    text_to_speech(chat_response)

    # Mostrar animación de respuesta
    result = pyfiglet.figlet_format("¡Respuesta!")
    print(result)

else:
    print("Debe configurar la variable de entorno OPENAI_API_KEY")
