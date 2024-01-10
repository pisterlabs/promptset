import openai
import speech_recognition as sr
from elevenlabs import generate, set_api_key, play, voices, Model, stream

print("Me encargare de traducir por voz al idioma ingles")

voices_list = voices()
#print(voices_list)

#Grabadora y Speech-to-text
def grabadora():
 
    r = sr.Recognizer()
 
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
  
        print("Te escucho...")
 
        audio = r.listen(source)
        
        # Reconocimiento de voz usando Google
        prompt = r.recognize_google(audio, language='es-ES')   #variable a enviar a CHATGPT para emprolijar y traducir.

        return prompt

#Por debajo config para CHATGPT
def ChatGPT_conversation(conversation):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation
    )

    #Configuraciones extras, no necesarias:
    # api_usage = response['usage']
    # print('Total token consumed: {0}'.format(api_usage['total_tokens']))
    # stop means complete
    # print(response['choices'][0].finish_reason)
    # print(response['choices'][0].index)
    
    conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
    return conversation

#Parametros Elevenlabs
set_api_key("YOUR ELEVENLABS API_KEY ")

#Parametros CHATGPT
api_key = "YOUR CHATGPT API KEY" #Con ChatGPT DaVinci 3.5 funciona bien, el contexto y traducción es correcto
openai.api_key = api_key
model_id = 'gpt-3.5-turbo'

prompt = "Comenzamos"
idioma_traducir = "ingles" #dar opcion a elegir al user

#ROLE PUede ser System / Assistant / user
conversation = []
conversation.append({'role': 'system', 'content': 'devolverme por respuesta solo el siguiente texto traducido al ' + idioma_traducir + ' y con las correcciones ortograficas que creas necesarias y SIN NOTAS: ' + prompt})
conversation = ChatGPT_conversation(conversation)
print('{0}: {1}\n'.format(conversation[-1]['role'].strip(), conversation[-1]['content'].strip()))

#Bucle principal para empezar la conversación continua
while True:
    #prompt = input('User:')
    prompt = grabadora()
    print("el prompt es: " + prompt)
    conversation.append({'role': 'user', 'content': 'devolverme por respuesta solo el siguiente texto (SIN NOTAS ADICIONALES) traducido al ' + idioma_traducir + ' y con las correcciones ortograficas que creas necesarias y SIN NOTAS: ' + prompt})
    conversation = ChatGPT_conversation(conversation)
    print('{0}: {1}\n'.format(conversation[-1]['role'].strip(), conversation[-1]['content'].strip()))
    texto_to_elevenlabs = (conversation[-1]['content'].strip() )

    #Generacion de audio opción #1
    audio = generate(
        text= texto_to_elevenlabs, #"Hi! I'm the world's most advanced text-to-speech system, made by elevenlabs.",
        voice="Antoni",
        model='eleven_monolingual_v1'
    )
    play(audio)

    #Generación de Audio opción #2
    #audio_stream = generate(
    #    text=texto_to_elevenlabs,
    #    stream=True,
    #    voice="Antoni",
    #    model='eleven_monolingual_v1'
    #)
    #stream(audio_stream)