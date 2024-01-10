import speech_recognition as sr
import os
from dotenv import load_dotenv
import sys
import time
import openai
from pydub import AudioSegment
from pydub.playback import play
import keyboard
import json
import traceback

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), 'keys.env'))
pkey = str(os.getenv("SERVICE_ACCOUNT_PKEY"))
id = str(os.getenv("SERVICE_ACCOUNT_ID"))
pkeyid = str(os.getenv("SERVICE_ACCOUNT_PKEYID"))
openai.organization = os.getenv("OPENAI_ORG")
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
openai.api_key = os.getenv("OPENAI_KEY")



logPath = "STTAITTSLog.txt"
recOn = False
x = 0
from google.oauth2 import service_account
import google.cloud.texttospeech as tts

dictt = {
  "type": "service_account",
  "project_id": "sttaitts",
  "private_key_id": pkeyid,
  "private_key": pkey,
  "client_email": "ghvgcfd@sttaitts.iam.gserviceaccount.com",
  "client_id": id,
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/ghvgcfd%40sttaitts.iam.gserviceaccount.com"
}





def text_to_wav(voice_name: str, text: str):
        try:
            with open('sdadsasg.json', 'w', encoding='utf-8') as f:
                json.dump(dictt, f, ensure_ascii=False, indent=2)
        except Exception as e:
            traceback.print_exception(e)
            wait_for_it = input('Press enter to close the terminal window')
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sdadsasg.json"

        language_code = "-".join(voice_name.split("-")[:2])
        text_input = tts.SynthesisInput(text=text)
        voice_params = tts.VoiceSelectionParams(
            language_code=language_code, name=voice_name
        )
        audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.LINEAR16)

        client = tts.TextToSpeechClient()
        response = client.synthesize_speech(
            input=text_input, voice=voice_params, audio_config=audio_config
        )
        os.remove("sdadsasg.json")

        filename = "tempTTS.wav"
        with open(filename, "wb") as out:
            out.seek(0)
            out.write(response.audio_content)
            print(f'Saved to: "{filename}"')
        out.close()


        clip = AudioSegment.from_wav(filename)
        play(clip)
        
mode = input("Choose a mode:\n  1 - Default\n   2 - F.R.I.D.A.Y\n   3 - GOOFY AHH\n")
aiMode = ""
if mode.lower() == "default" or mode.lower() == "1":
    aiMode = "Jsi užitečný assistant."
elif mode.lower() == "friday" or mode.lower() == "f.r.i.d.a.y" or mode.lower() == "2":
    aiMode = "Jsi F.R.I.D.A.Y z filmu Infinity War."
elif mode.lower() == "goofy ahh" or mode.lower() == "goofy" or mode.lower() == "3":
    aiMode = "Jsi borka co pořád říká fr fr, bez čepice, na boha, ledžit"
else:
    aiMode = "Jsi užitečný assistant."


mess = [
    {"role": "system", "content": aiMode}
    ]

while True:
    if not recOn:
        try:
            if keyboard.is_pressed(';'): 
                recOn = True
                pass
        except:
            pass

    if recOn:
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    text_to_wav("cs-CZ-Wavenet-A", "Ano?")
                    print("Ano?")
                    audio = r.listen(source)
                try:
                    saidString = r.recognize_google(audio, language='cs-CZ')
                    print("Recognized: " + saidString)

                    if saidString.lower() == "what's the temperature" or saidString.lower() == "what is the temperature" or saidString.lower() == "jaká je teplota" or saidString.lower() == "teplota":
                        gpt_prompt = "Jaká je v Černošicích teplota?"
                        response = openai.Completion.create(
                        engine="text-davinci-003",
                        prompt=gpt_prompt,
                        temperature=0.5,
                        max_tokens=256,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                        )
                        print(response['choices'][0]['text'])

                        f = open(logPath, "w", encoding="utf-8")
                        f.write(response['choices'][0]['text']+"\n")
                        f.close()

                        text_to_wav("cs-CZ-Wavenet-A", response['choices'][0]['text'])

                    elif saidString.lower() == "what's the time" or saidString.lower() == "what is the time" or saidString.lower() == "what time is it" or saidString.lower() == "čas" or saidString.lower() == "kolik je hodin" or saidString.lower() == "kolik je":
                        
                        t = time.localtime()
                        TIME = time.strftime("%I:%M %p", t)

                        print(f"Je {TIME}.")

                        f = open(logPath, "w", encoding="utf-8")
                        f.write(f"Je {TIME}."+"\n")
                        f.close()

                        text_to_wav("cs-CZ-Wavenet-A", f"Je {TIME}.")
                    
                    elif saidString.lower() == "open steam" or saidString.lower() == "otevři steam" or saidString.lower() == "steam":
        
                        print("Otevírám Steam.")

                        os.startfile("C:\\Program Files (x86)\\Steam\\steam.exe")

                        f = open(logPath, "w", encoding="utf-8")
                        f.write("Otevírám Steam.")
                        f.close()

                        text_to_wav("cs-CZ-Wavenet-A", "Otevírám Steam.")

                    elif saidString.lower() == "shut down" or saidString.lower() == "turn off" or saidString.lower() == "shutdown" or saidString.lower() == "vypnout" or saidString.lower() == "vypni se" or saidString.lower() == "konec":

                        resp = "Vypínám se."
                        print(resp)

                        f = open(logPath, "w", encoding="utf-8")
                        f.write(resp)
                        f.close()

                        text_to_wav("cs-CZ-Wavenet-A", resp)

                        sys.exit()

                    elif saidString.lower() == "bedtime" or saidString.lower() == "turn off my computer" or saidString.lower() == "goodnight" or saidString.lower() == "good night" or saidString.lower() == "dobrou" or saidString.lower() == "dobrou noc" or saidString.lower() == "jdu spát" or saidString.lower() == "vypni mi počítač" or saidString.lower() == "vypni počítač":

                        if saidString.lower() == "good night" or saidString.lower() == "bedtime" or saidString.lower() == "goodnight" or saidString.lower() == "dobrou" or saidString.lower() == "dobrou noc" or saidString.lower() == "jdu spát":
                            resp = "Dobrou noc, vypnu vám počítač."
                        else:
                            resp = "Vypínám vám počítač."
                        print(resp)

                        f = open(logPath, "w", encoding="utf-8")
                        f.write(resp)
                        f.close()

                        text_to_wav("cs-CZ-Wavenet-A", resp)

                        os.system("shutdown /s /t 1")

                    elif saidString.lower() == "go to sleep" or saidString.lower() == "put my computer to sleep" or saidString.lower() == "jdi spát" or saidString.lower() == "dej můj počítač do spacího režimu":

                        
                        resp = "Dávám váš počítač do spacího režimu."
                        print(resp)

                        f = open(logPath, "w", encoding="utf-8")
                        f.write(resp)
                        f.close()

                        text_to_wav("cs-CZ-Wavenet-A", resp)

                        os.system("Rundll32.exe Powrprof.dll,SetSuspendState Sleep") 
                    else:
                        mess.append({"role" : "user", "content" : saidString})
                        response = openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=mess
                        )
                        
                        mess.append({"role" : "assistant", "content" : response['choices'][0]['message']['content']})
                        print(mess)
                        print(response['choices'][0]['message']['content'])

                        f = open(logPath, "w", encoding="utf-8")
                        f.write(response['choices'][0]['message']['content']+"\n")
                        f.close()

                        if response['choices'][0]['message']['content'][0] == "?":
                            text_to_wav("cs-CZ-Wavenet-A", response['choices'][0]['message']['content'][1:])
                        else:
                            text_to_wav("cs-CZ-Wavenet-A", response['choices'][0]['message']['content'])
                except sr.UnknownValueError:
                    print("Could not understand audio")



                    
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
            

            
            

    
    recOn = False
    os.system('cls' if os.name == 'nt' else 'clear')