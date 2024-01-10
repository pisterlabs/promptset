import speech_recognition as sr
import openai

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

# Cria um objeto de reconhecimento
r = sr.Recognizer()

# Usa o microfone como fonte de entrada de áudio
with sr.Microphone() as source:
    print("Fale alguma coisa:")
    # Aguarda o usuário falar algo e grava o áudio
    audio = r.listen(source)

# Reconhece a fala usando o Google Web Speech API
try:
    text = r.recognize_google(audio, language='pt-BR')
    print("Você disse: " + text)

    # Passa o texto reconhecido para a API do OpenAI
    response = generate_response(text)
    print(response)
    
except sr.UnknownValueError:
    print("Não entendi o que você disse")
except sr.RequestError as e:
    print("Não foi possível se conectar ao servidor do Google Speech Recognition; {0}".format(e))
