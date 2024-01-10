import pyaudio
import wave
import threading
import os
import openai
from gtts import gTTS
from playsound import playsound
from openai import OpenAI
import speech_recognition as sr

client = OpenAI()

# Initialize the API key
def gravar_audio(nome_arquivo, taxa_amostragem=44100, n_canais=1, largura_amostra=pyaudio.paInt16):
    """
    Grava áudio do microfone e salva em um arquivo WAV. A gravação continua até que o usuário pressione 'Enter'.

    :param nome_arquivo: Nome do arquivo WAV a ser criado.
    :param taxa_amostragem: Taxa de amostragem do áudio (em Hz).
    :param n_canais: Número de canais (1 para mono, 2 para estéreo).
    :param largura_amostra: Formato da amostra de áudio (pyaudio.paInt16 é comum).
    """

    p = pyaudio.PyAudio()  # Cria uma interface PyAudio

    # Abrindo stream para gravação
    stream = p.open(format=largura_amostra, channels=n_canais, rate=taxa_amostragem, input=True, frames_per_buffer=1024)

    frames = []  # Lista para armazenar os frames capturados

    def gravar():
        print("Iniciando gravação. Pressione 'Enter' para parar...")
        while not input_event.is_set():
            data = stream.read(1024)
            frames.append(data)

    input_event = threading.Event()
    recording_thread = threading.Thread(target=gravar)
    recording_thread.start()

    input("Pressione 'Enter' para parar a gravação...\n")
    input_event.set()

    recording_thread.join()

    # Parando e fechando o stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Salvando os dados em um arquivo WAV
    with wave.open(nome_arquivo, 'wb') as wf:
        wf.setnchannels(n_canais)
        wf.setsampwidth(p.get_sample_size(largura_amostra))
        wf.setframerate(taxa_amostragem)
        wf.writeframes(b''.join(frames))

    print("Gravação finalizada.")

# Exemplo de uso
nome_arquivo = 'audio_gravado.wav'
idioma = 'pt'


def gerar_resposta(messages):
    #response = openai.ChatCompletion.create( ## Api antiga
    response = openai.chat.completions.create( ## API nova
        model="gpt-3.5-turbo", 
        messages=messages,
        temperature=0.5
    )
    return [response.choices[0].message.content, response.usage]

mensagens = [{"role": "system", "content": "assistant_name = SARA,attributes =eficiente"}]


def converter_audio_em_texto(nome_arquivo):    

    # Inicializa o reconhecedor de fala
    r = sr.Recognizer()
    

    # Carrega o arquivo de áudio
    with sr.AudioFile(nome_arquivo) as source:
        audio_data = r.record(source)

    # Tenta reconhecer o fala usando o Google Web Speech API
    try:
        texto = r.recognize_google(audio_data, language='pt-BR')
        print("Texto transcrito: " + texto)
        return texto
    except sr.UnknownValueError:
        print("Google Speech Recognition não conseguiu entender o áudio.")
    except sr.RequestError as e:
        print(f"Não foi possível solicitar resultados do serviço Google Speech Recognition; {e}")


while True:
    # Ask a question
    question = input("Perguntar para Sara (\"sair\"): ")

    if question == "sair" :
        print("saindo")
        break
    elif question == "":
        gravar_audio(nome_arquivo)

        texto_transcrito = converter_audio_em_texto(nome_arquivo)     

        mensagens.append({"role": "user", "content": str(texto_transcrito)})
        answer = gerar_resposta(mensagens)
        print("user:", question)
        print("Sara:", answer[0], "\nCusto:\n", answer[1])
        mensagens.append({"role": "assistant", "content": answer[0]})
        tts = gTTS(text=answer[0], lang=idioma, slow=False)
        # Salvando o arquivo de áudio
        arquivo_audio = "audio.mp3"
        tts.save(arquivo_audio)
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=answer[0],
        )

        response.stream_to_file(arquivo_audio)

        # Reproduzindo o arquivo de áudio
        playsound(arquivo_audio)

        # Opcional: remover o arquivo de áudio após a reprodução
        os.remove(arquivo_audio)
    else:
        mensagens.append({"role": "user", "content": str(question)})

        answer = gerar_resposta(mensagens)
        print("user:", question)
        print("Sara:", answer[0], "\nCusto:\n", answer[1])
        mensagens.append({"role": "assistant", "content": answer[0]})
        # tts = gTTS(text=answer[0], lang=idioma, slow=False)
        # Salvando o arquivo de áudio
        arquivo_audio = "audio.mp3"
        # tts.save(arquivo_audio)
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=answer[0],
        )

        response.stream_to_file(arquivo_audio)

        # Reproduzindo o arquivo de áudio
        playsound(arquivo_audio)
        
        # Opcional: remover o arquivo de áudio após a reprodução
        os.remove(arquivo_audio)
    debugar = False
    if debugar:
        print("Mensagens", mensagens, type(mensagens))



