import pyttsx3, playsound,os
import cv2
import _thread
from ManLypsinc import linearize
from gtts import gTTS
from transformers import pipeline
import speech_recognition as sr
model_inp=input("Enter model {pyttsx3, whisper, gtts}: ")

#Voice Speech to text
def input_STT()->str:
    recognizer=sr.Recognizer()
    print("Speak a sentence.")
    while True:
        with sr.Microphone() as mic:
            print("Listening, pls go ahead... ")
            recognizer.adjust_for_ambient_noise(mic, duration=1)
            audio=recognizer.listen(mic)
        try:
            text=recognizer.recognize_google(audio)
            print("Recongnised Voice: Converting Speech to text ...")
            return text

        except Exception as e :
            if e == sr.UnknownValueError():
                print("Coudn't understand audio. PLease try again")
                return None
            elif e == sr.RequestError():
                print("Unable to access speech recognition service")
                return None
            else:
                print(e)
                return None
input_sent=input_STT()

# Model calling

classifier = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')   #Here sentiment-analysis is the model name, pipeline(...) does all your work
result =classifier(input_sent)
print(result)
print(result[0]['label'])
text=result[0]['label']

meta_text=''
text='it is a '+result[0]['label']+' sentence'
waitempty, waitfullstop, waitchar=0,0,0
def play_audio_pyttsx3(text):
    engine = pyttsx3.init()
    print(text)
    #change rate of speakinga
    engine.setProperty('rate', 20)   
    engine.say(text)
    engine.runAndWait()
def play_audio_gtts(text):
    tts = gTTS(text=text, lang='en')  #gtts has default speed, cant change
    tts.save("tts.mp3")
    playsound.playsound("tts.mp3") 
    os.remove("tts.mp3")
def play_audio_whisper(text:str):
    from openai import OpenAI
    # Make sure you do $pip install openai==1.3.5 ONLY
    OPENAI_API_KEY = "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"
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.audio.speech.create(
        model="tts-1-hd",   # "tts-1", "tts-1-hd" are 2 models
        voice="onyx",    # alloy, echo, fable, onyx, nova, and shimmer. Choose any of these voices.
        speed=1.0,  # speed varies betwen 0.0 to 4.0. 1.0 is default
        response_format="mp3", # mp3, opus, aac, and flac are supported audio formats.
        input=text,
    )
    response.stream_to_file("output.mp3")
    # playsound "output.mp3"
    playsound.playsound("output.mp3")
    os.remove("output.mp3")

if model_inp=='gtts':
    _thread.start_new_thread( play_audio_gtts, (text,) )
    meta_text='         '+linearize(text)+'                                    '
    print(meta_text)
    waitchar,waitfullstop,waitempty=65,21,21

elif model_inp=='pyttsx3':
    _thread.start_new_thread( play_audio_pyttsx3, (text,) )
    meta_text='         '+linearize(text)+'                                       '
    print(meta_text)
    waitchar,waitfullstop,waitempty=90,31,60 # for rate=20, manually set for other rates.

else:
    _thread.start_new_thread( play_audio_whisper, (text,) )
    meta_text='                                                             ' + linearize(text)+'                        '
    print(meta_text)
    waitchar,waitfullstop,waitempty=60,10,20  # for default speed=1.0, manually set for other speeds.
    
mouthdict={
    ' ':'nothing.png',
    '.':'nothing.png',
    '!':'nothing.png',
    '?':'nothing.png',
    ',':'nothing.png',
    ', ':'nothing.png',
    'a':'a.png',
    'b':'b.png',
    'c':'c.png',
    'd':'d.png',
    'e':'e.png',
    'f':'f.png',
    'g':'g.png',
    'h':'h.png',
    'i':'i.png',
    'j':'j.png',
    'k':'k.png',
    'l':'l.png',
    'm':'m.png',
    'n':'n.png',
    'o':'o.png',
    'p':'p.png',
    'q':'q.png',
    'r':'r.png',
    's':'s.png',
    't':'t.png',
    'u':'u.png',
    'v':'v.png',
    'w':'w.png',
    'x':'x.png',# for ch, sh. x.png is fine
    'y':'y.png',
    'z':'z.png',  
    '0':'t.png'   # for th
}
for char in meta_text:
    if char in mouthdict:
        if char==' ':
            mouth=mouthdict[char]
            print(mouth)
            cv2.imshow(f"{char}", cv2.imread("/Volumes/Anirudh/VS_Code/LipSyncing/SpeechMouthMan/"+mouth))
            cv2.waitKey(waitempty) 
            # 20-22 for gtts, en   # 30 for engine rate=30 pyttsx3, 60 for rate=20, # 20 for whisper=1.0
            cv2.destroyAllWindows()
        elif char in ['.','!','?']:
            mouth=mouthdict[char]
            print(mouth)
            cv2.imshow(f"{char}", cv2.imread("/Volumes/Anirudh/VS_Code/LipSyncing/SpeechMouthMan/"+mouth))
            cv2.waitKey(waitfullstop) 
            # 20-22 for gtts, en   # 30 for engine rate=30 pyttsx3, 31 for rate=20 # 11 for whisper=1.0
            cv2.destroyAllWindows()
        elif char in [',',', ']:
            mouth=mouthdict[char]
            print(mouth)
            cv2.imshow(f"{char}", cv2.imread("/Volumes/Anirudh/VS_Code/LipSyncing/SpeechMouthMan/"+mouth))
            cv2.waitKey(waitfullstop*8) 
            # 20-22 for gtts, en   # 30 for engine rate=30 pyttsx3, 31 for rate=20 # 11 for whisper=1.0
        else:
            mouth=mouthdict[char]
            print(mouth)
            cv2.imshow(f"{char}", cv2.imread("/Volumes/Anirudh/VS_Code/LipSyncing/SpeechMouthMan/"+mouth))
            cv2.waitKey(waitchar)
            # 65 for gtts, en  # 90 for engine rate=30 pyttsx3, 90 for rate=20 # 65 for whisper=1.0
            # 65 for hi,ja(most languages)
            cv2.destroyAllWindows()