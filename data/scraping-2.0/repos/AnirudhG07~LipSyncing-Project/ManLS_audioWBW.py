import pyttsx3, playsound,os, time
import cv2
import _thread
from ManLypsinc import linearize
from gtts import gTTS

text=input("Enter text: ")

def play_audio_pyttsx3(text):
    engine = pyttsx3.init()
    #print(text)
    #change rate of speaking
    engine.setProperty('rate', 20)   
    engine.say(text)
    engine.runAndWait()
def play_audio_gtts(text):
    for word in text.split():
        tts = gTTS(text=word, lang='en')  #gtts has default speed, cant change
        tts.save("tts.mp3")
        playsound.playsound("tts.mp3")
        os.remove("tts.mp3")
def play_audio_whisper(text:str):
    from openai import OpenAI
    # Make sure you do $pip install openai==1.3.5 ONLY
    OPENAI_API_KEY = "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"
    client = OpenAI(api_key=OPENAI_API_KEY)
    for word in text.split():
        response = client.audio.speech.create(
            model="tts-1-hd",   # "tts-1", "tts-1-hd" are 2 models
            voice="onyx",    # alloy, echo, fable, onyx, nova, and shimmer. Choose any of these voices.
            speed=1.2,  # speed varies betwen 0.0 to 4.0. 1.0 is default
            response_format="mp3", # mp3, opus, aac, and flac are supported audio formats.
            input=word,
        )
        response.stream_to_file("output.mp3")
        # playsound "output.mp3"
        playsound.playsound("output.mp3")
        os.remove("output.mp3")

model_inp=input("Enter model {pyttsx3, whisper, gtts}: ")
if model_inp=='gtts':
    _thread.start_new_thread( play_audio_gtts, (text,) )
    meta_text='         '+linearize(text)+'   '
    print(meta_text)
elif model_inp=='pyttsx3':
    print("Sorry this program is not working with pyttsx3. Retry for other models")
    exit()
else:
    _thread.start_new_thread( play_audio_whisper, (text,) )
    meta_text=' '+ linearize(text)+' '
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
    'x':'x.png', # for sh, ch
    'y':'y.png',
    'z':'z.png',
    '0':'t.png'   # for th
}

for i in range(12):
    mouth=mouthdict[' ']
    #print(mouth)
    cv2.imshow(f"{' '}", cv2.imread("/Volumes/Anirudh/VS_Code/LipSyncing/SpeechMouthMan/"+mouth))
    cv2.waitKey(23)  # 20 for gtts, # 23 for whisper
for char in meta_text:
    if char in mouthdict: 
        if char==' ' or char=='.': 
            mouth=mouthdict[char]
            #print(mouth)
            cv2.imshow(f"{char}", cv2.imread("/Volumes/Anirudh/VS_Code/LipSyncing/SpeechMouthMan/"+mouth))
            cv2.waitKey(1500) 
            # 700 for gtts, #doesn't work for pyttsx3 # 1500 for whisper
            cv2.destroyAllWindows()
        else:
            mouth=mouthdict[char]
            #print(mouth)
            cv2.imshow(f"{char}", cv2.imread("/Volumes/Anirudh/VS_Code/LipSyncing/SpeechMouthMan/"+mouth))
            cv2.waitKey(78)
            # 65 for gtts, en  # 78 for whisper
            # 65 for hi,ja(most languages)
            cv2.destroyAllWindows()
