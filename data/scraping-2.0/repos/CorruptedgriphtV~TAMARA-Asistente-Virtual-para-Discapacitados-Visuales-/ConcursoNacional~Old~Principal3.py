
import cv2
import requests
import pyttsx3
import openai
import os
import pygame
#Instala todo con pip install opencv-python requests pyttsx3 openai pygame
import speech_recognition as sr
import pyaudio


def play_sound(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
def generate_text(prompt):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100
    )
    #print("Respuesta de GPT-3:", response)  # Agrega esta línea para ver la respuesta en bruto
    generated_text = response.choices[0].text.strip()
    #print("Texto generado:", generated_text)  # Agrega esta línea para ver el texto generado
    return generated_text

api_key = "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"
openai.api_key = api_key

def main():
    #for i, microphone_name in enumerate(sr.Microphone.list_microphone_names()):
        #print(i, microphone_name)
    # Configurar las credenciales de la API de GPT-3
    # Inicializar el reconocedor de voz
    r = sr.Recognizer()


    # Configurar la URL y las credenciales de la API de Computer Vision
    endpoint = "https://tamaraserver.cognitiveservices.azure.com/"
    subscription_key = "975da0dae182426ba219f03767e152ae"
    api_url = f"{endpoint}vision/v3.2/analyze"

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    while True:
 
        # Capturar la imagen desde la cámara
        ret, frame = cap.read()
        # Reproducir sonido intro antes de escuchar la solicitud
       

        # Guardar la imagen en un archivo temporal
        img_temp_path = "temp_img.jpg"
        cv2.imwrite(img_temp_path, frame)

        # Leer la imagen como bytes
        with open(img_temp_path, "rb") as image_file:
            image_data = image_file.read()

        # Realizar la solicitud a la API de Computer Vision
        headers = {
            "Content-Type": "application/octet-stream",
            "Ocp-Apim-Subscription-Key": subscription_key
        }
        params = {
            "visualFeatures": "Description",
            "language": "es"  # Solicitar descripción en español
        }
        response = requests.post(api_url, headers=headers, params=params, data=image_data)

        # Procesar la respuesta de la API de Computer Vision
        if response.status_code == 200:
            result = response.json()
            if "description" in result and "captions" in result["description"]:
                description = result["description"]["captions"][0]["text"]

                # Guardar la descripción en un archivo temporal en la carpeta Sandbox
                with open("temp_description.txt", "w") as f:
                    f.write(description)
            else:
                print("No se encontró una descripción en la respuesta de Azure.")
        else:
            print("Error en la solicitud a Azure:", response.text)

        # Eliminar el archivo temporal de la imagen
        if os.path.exists(img_temp_path):
            os.remove(img_temp_path)

        # Esperar la entrada del usuario
       
        with sr.Microphone(device_index=2) as source:
            print("Esperando el comando de activación...")
            audio = r.listen(source, timeout=5)

        try:
            user_input = r.recognize_google(audio, language='es-ES')

            # Si el usuario dijo "Oye TAMARA", empezar a procesar el audio
            if "Oye TAMARA" in user_input:
                print("Dime algo:")
                play_sound("Intro.mp3")
                with sr.Microphone(device_index=13) as source:
                    audio = r.listen(source, timeout=5)

                try:
                    user_input = r.recognize_google(audio, language='es-ES')
                    print("Creo que dijiste: " + user_input)
                except sr.UnknownValueError:
                    print("Google Speech Recognition no entendió lo que dijiste")
                except sr.RequestError as e:
                    print("No se pudo solicitar resultados a Google Speech Recognition; {0}".format(e))
        except sr.UnknownValueError:
            print("Google Speech Recognition no entendió el comando de activación")
        except sr.RequestError as e:
            print("No se pudo solicitar resultados a Google Speech Recognition; {0}".format(e))

        

        # Reproducir sonido outro después de recibir la solicitud
        play_sound("Outro.mp3")

        # Si el usuario pregunta qué está viendo o qué se ve
        if any(phrase in user_input.lower() for phrase in ["qué estoy viendo", "qué se ve", "que estoy viendo", "que se ve"]):
            # Hablar la descripción generada por Azure
            engine = pyttsx3.init()
            engine.setProperty('voice', 'spanish')  # Establecer idioma a español
            engine.say("Imagen reconocida: " + description)
            engine.runAndWait()

        # Generar respuesta adicional con GPT-3 utilizando la descripción almacenada en el archivo temporal
        with open("temp_description.txt", "r") as f:
            saved_description = f.read()
        prompt = "Eres TAMARA, asistes a personas con discapacidad visual y tu objetivo es proporcionar una descripción detallada de la imagen y ayudar en lo que el usuario solicite, no inventes muchas cosas, solo retroalimenta a lo que ves .\n\nUsuario: {}\n\nDescripción de Azure: {}\n\nRespuesta:"
        prompt_with_input = prompt.format(user_input, saved_description)
        additional_assistance = generate_text(prompt_with_input)

        # Imprimir y pronunciar la respuesta adicional generada por GPT-3
        print("Respuesta adicional generada por GPT-3:", additional_assistance)
        engine = pyttsx3.init()
        engine.setProperty('voice', 'spanish')  # Establecer idioma a español
        engine.say(additional_assistance)
        engine.runAndWait()

if __name__ == "__main__":
    main()