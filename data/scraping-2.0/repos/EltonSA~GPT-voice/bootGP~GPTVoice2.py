import speech_recognition as sr
import openai
import pyttsx3

openai.api_key = "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"

def generate_response(prompt):
    model_engine = "text-davinci-002"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = response.choices[0].text
    return message.strip()

def speak(text):
    # Inicializa o objeto da engine de fala
    engine = pyttsx3.init()
    # Define a velocidade de fala (padrão é 200)
    engine.setProperty('rate', 150)
    # Define o volume de fala (padrão é 1.0)
    engine.setProperty('volume', 1.0)
    # Usa a voz padrão do sistema
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    # Fala o texto fornecido
    engine.say(text)
    engine.runAndWait()

# Defina a palavra de ativação
activation_phrase = "olá Jarvis, o papai chegou"

input_type = input("Como você deseja fornecer a entrada? (digite 'text' ou 'voice'): ")

if input_type == "voice":
    # Cria um objeto de reconhecimento
    r = sr.Recognizer()

    while True:
        # Usa o microfone como fonte de entrada de áudio
        with sr.Microphone() as source:
            print("Fale alguma coisa:")
            # Aguarda o usuário falar algo e grava o áudio
            audio = r.listen(source)

        try:
            # Reconhece a fala usando o Google Web Speech API
            text = r.recognize_google(audio, language='pt-BR')
            print("Você disse: " + text)

            # Verifica se a palavra de ativação foi dita
            if activation_phrase.lower() in text.lower():
                # Passa o texto reconhecido para a API do OpenAI
                response = generate_response(text)
                print(response)

                # Fala a resposta
                speak(response)

                break
        
        except sr.UnknownValueError:
            print("Não entendi o que você disse. Por favor, tente novamente.")
            speak("Não entendi o que você disse. Por favor, tente novamente.")
        
        except sr.RequestError as e:
            print("Não foi possível se conectar ao servidor do Google Speech Recognition; {0}".format(e))
            speak("Não foi possível se conectar ao servidor do Google Speech Recognition. Por favor, tente novamente.")
else:
    prompt = input("Digite sua pergunta: ")
    response = generate_response(prompt)
    print(response)
    speak(response)
