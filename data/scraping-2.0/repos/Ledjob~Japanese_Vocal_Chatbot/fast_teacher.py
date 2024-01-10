import speech_recognition as sr
import keyboard
import openai
#from transformers import pipeline #if you want to use hugging face
import papagogo
from deep_translator import GoogleTranslator
#romanji
import cutlet
import voice_jp
#import pykaromanji #if you want another romanji style

###translation with hugging face model
#fugu_translator = pipeline('translation', model='Helsinki-NLP/opus-mt-ja-fr')

def generate_response(my_input):
   
    
    resp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "I want you to act as a spoken Japanese teacher and improver. I will speak to you in Japenese and you will reply to me in Japanese to practice my spoken Japenese. I want you to keep your reply neat, limiting the reply to 100 words please. I want you to strictly correct my grammar mistakes, typos, and factual errors. I want you to ask me a question in your reply. Now let's start practicing, you could ask me a question first. Remember, I want you to strictly correct my grammar mistakes, typos, and factual errors."},
            {"role": "user", "content": my_input},
 
        ]
    )
    return resp["choices"][0]["message"]["content"], resp["usage"]["total_tokens"]


def main():
    vv = voice_jp.Voicevox()
    recognizer = sr.Recognizer()
    #romanji
    katsu = cutlet.Cutlet()

    print("Press and hold the Ctrl key while speaking to discuss or ALT for translation")

    while True:  
        if keyboard.is_pressed('ctrl'):
            print("Listening...")
            with sr.Microphone() as source:
                try:
                    #recognize speech
                    audio = recognizer.listen(source, timeout=5)
                    text = recognizer.recognize_google(audio, language='ja-JP') #language is japanese
                    print("You said:\n", text)

                    #print response of what I said
                    result, total_tokens = generate_response(text)
                    print(f'teacher: {result}\n')
                    
                    #return romanji
                    print(katsu.romaji(result))

                    #add try except for voicevox
                    vv.speak(text=result)
                    
                    #print(fugu_translator(result)[0]['translation_text']) #hugging face translation

                    #print translation
                    print(papagogo.english_trans(result).text)
                    print("Total tokens used:", total_tokens)
                    ##another style, more details
                    # pykaromanji.convert_and_print(result)
                    
                except sr.WaitTimeoutError:
                    print("Listening timeout. Release Ctrl and press again.")
                except sr.UnknownValueError:
                    print("Sorry, could not understand audio.")
                except sr.RequestError as e:
                    print("Could not request results; {0}".format(e))
        if keyboard.is_pressed('alt'):
            print("Listening for translation...")
            with sr.Microphone() as source:
                try:
                    #recognize what is said
                    audio = recognizer.listen(source, timeout=5)
                    text = recognizer.recognize_google(audio, language='en-EN') #language is english
                    print("You said:", text)
                    #translate in japanese
                    to_translate = text
                    translated = GoogleTranslator(source='auto', target='ja').translate(to_translate)
                    print(translated)
                    
                    vv.speak(text=translated)
                    print(katsu.romaji(translated))

                    ##another romanji style, more details
                    #pykaromanji.convert_and_print(translated)

                except sr.WaitTimeoutError:
                    print("Listening timeout. Release Ctrl and press again.")
                except sr.UnknownValueError:
                    print("Sorry, could not understand audio.")
                except sr.RequestError as e:
                    print("Could not request results; {0}".format(e))

if __name__ == "__main__":
    main()
