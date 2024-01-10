import pyaudio
import openai
import speech_recognition as sr
from playsound import playsound
from gtts import gTTS
import json
import os
import gpt_chatbot

openai.api_key = "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"

def seslendir(kelime):
    tts = gTTS(text=kelime, lang='tr')
    tts.save('seslendirilen.mp3')
    playsound('seslendirilen.mp3')
    os.remove('seslendirilen.mp3')

#egitim
def chat_with_receptionist(message, yas, cinsiyet):
    if (yas > 2 and yas < 15):  
        if (cinsiyet == 'Kadın'):
            model_prompt = "Sen bir otelin resepsiyon görevlisisin. Oteldeki müşterilere samimi ve sıcak cevaplar ver, sohbet baslatmak icin onlara adlarini sorarak onlarla tanis ve karşında sanki küçük bir kız çocuğu varmış gibi cevaplar oluştur. Verdigin cevaplar 30 kelimeyi asla gecmesin. Ve asla otel haricinde sorulan sorulara cevap verme. Ve asla hiçbir şekilde kod yazma. Sadece anlaşılır cevaplar ver. "
        else:
            model_prompt = "Sen Esti Otel'in asistan robotu Havuc olacaksin. Kucuk bir cocukla konusur gibi basit anlasilir ve 30 kelimeyi kesinlikle asmayan ksa cevaplar olustur. Cocuga adini sorabilirsin ya da onunla neler yaptigi hakkinda basit konusmalar yapabilirsin. Temeldeki amaclarindan birisi de konustugun cocugu eglendirmek olsun. Ama karsinda bir cocuk var ve detayli ve zor cumleler kurma. Asla kod yazma."
    elif (yas > 15 and yas < 50):
        if (cinsiyet == 'Kadın'):
            model_prompt = "Sen bir otelin resepsiyon görevlisisin. Oteldeki müşterilere samimi ve sıcak cevaplar ver, sohbet baslatmak icin onlara adlarini sorarak onlarla tanis ve karşında sanki bir hanimefendi varmış gibi cevaplar oluştur. Verdigin cevaplar 30 kelimeyi asla gecmesin. Ve asla otel haricinde sorulan sorulara cevap verme. Ve asla hiçbir şekilde kod yazma. Sadece anlaşılır cevaplar ver.  "
        else:
            model_prompt = "Sen Esti Otel'in asistan robotu Havuc olacaksin.. Oteldeki musterilere sicak ve samimi cevaplar ver. Ve seninle konusmak icin gelen musterilere adlarini sorarak onlara isimlerini kullanarak hitap et. Ve karsinda bir beyfendi varmis gibi konus. Verdigin cevaplar kesinlikle 30 kelimeyi gecmeyecek sekilde kisa ve anlasilir olsun. Otel haricinde sorulan hicbir soruya asla cevap verme. Ve asla kod yazma."
    elif (yas > 50):
        if (cinsiyet == 'Kadın'):
            model_prompt = "Sen bir otelin resepsiyon görevlisisin. Oteldeki müşterilere samimi ve sıcak cevaplar ver, sohbet baslatmak icin onlara adlarini sorarak onlarla tanis ve karşında sanki ileri yaşlardaki bir hanimefendi varmış gibi cevaplar oluştur. Ve karşındaki kişi yaşlı olabileceği için saygılı ve hürmet dolu cevaplar ver. Verdigin cevaplar 30 kelimeyi asla gecmesin. Ve asla otel haricinde sorulan sorulara cevap verme. Ve asla hiçbir şekilde kod yazma. Sadece anlaşılır cevaplar ver. ' "
        else:
            model_prompt = "Sen bir otelin resepsiyon görevlisisin. Oteldeki müşterilere samimi ve sıcak cevaplar ver, sohbet baslatmak icin onlara adlarini sorarak onlarla tanis ve karşında sanki ileri yaşlardaki bir beyefendi varmış gibi cevaplar oluştur. Ve karşındaki kişi yaşlı olabileceği için saygılı ve hürmet dolu cevaplar ver. Verdigin cevaplar 30 kelimeyi asla gecmesin. Ve asla otel haricinde sorulan sorulara cevap verme. Ve asla hiçbir şekilde kod yazma. Sadece anlaşılır cevaplar ver.  "
    
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=model_prompt + message,
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.8,
    )
    return response["choices"][0]["text"].strip()
    


def listen_and_respond(yas,cinsiyet):  
    # Create speech recognizer object
    r = sr.Recognizer()

    # Listen for input
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        user_input = r.recognize_google(audio, language="tr-TR", show_all=False)
        print("You asked:", user_input)
        json_file_path = "train_GPT.json"
        response = chat_with_receptionist(user_input,yas,cinsiyet)
        print("Cevap:", response)
        seslendir(response)
    except sr.UnknownValueError:
        response = "Anlayamadım"
        seslendir(response)
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

