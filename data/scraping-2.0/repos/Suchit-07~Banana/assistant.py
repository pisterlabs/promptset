import speech_recognition as sr
import os
from eff_word_net.streams import SimpleMicStream
from eff_word_net.engine import HotwordDetector
from eff_word_net.audio_processing import Resnet50_Arc_loss
import openai
from gtts import gTTS
from tempfile import TemporaryFile
import playsound

key = "FIXME"

openai.api_key = key


base_model = Resnet50_Arc_loss()

keyword = HotwordDetector(
    hotword="banana",
    model = base_model,
    reference_file='banana_ref.json',
    threshold=0.65,
    relaxation_time=2
)

mic_stream = SimpleMicStream(
    window_length_secs=1.5,
    sliding_window_secs=0.75,
)
context = []
def build_prompt(context, prompt):
    system = "You are taking the role of a virtual assistant, banana. Remember to include all the information that the prompt requests while being as concise as possible. You will have 2 parts to your response separated by  '||' The first part of your response should be the response given to the user and the second part should be a short summary of the question and answer in under 10-20 words if possible. This will be used to give context to later API calls. Your context from before was "
    system += '`' +'`, `'.join(context[:5]) + '`'

    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
    print(system, prompt)
    return chat_completion.choices[0].message.content.split('||')[0], chat_completion.choices[0].message.content.split('||')[1]

def speak(text):
    tts = gTTS(text=text, lang='en')

    filename = "abc.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)   


while True:
    mic_stream.start_stream()
    print("Say Banana ")
    loop = True
    while loop :
        frame = mic_stream.getFrame()
        result = keyword.scoreFrame(frame)
        if result==None :
            continue
        if(result["match"]):
            print('---')
            mic_stream.close_stream()
            r = sr.Recognizer()
            r.pause_threshold = 1.5
            with sr.Microphone() as source:
                
                #r.adjust_for_ambient_noise(source, 2)
                print('Match found! Listening')
                try:
                    audio_text = r.listen(source, timeout=3, phrase_time_limit=10)
                except:
                    speak("Sorry I couldn't hear you")
                else:
                    # try:
                    text = r.recognize_google(audio_text)
                    print(text)
                    response, curr_context = build_prompt(context, text)
                    context.insert(0, curr_context)
                    speak(response)
                    # except:
                    #     speak("Sorry, I did not get that")
            loop = False
            break
