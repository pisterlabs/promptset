import openai
from youtube_processor import youtube_preprocess
from video_chunk import chunk_by_size
import time
import os
import dotenv

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

# Solicita o link do vídeo ao usuário
link = input("Enter the link of the video: ")

# Marca o início do tempo de execução do código
inicio = time.time()

# Pré-processa o vídeo do YouTube, extraindo o áudio e o título do vídeo
audio_file, video_title = youtube_preprocess(link)

# Divide o áudio do vídeo em trechos
no_of_chunks = int(chunk_by_size(audio_file, video_title))

# Transcreve cada trecho do áudio usando a API do OpenAI Whisper
for i in range(no_of_chunks + 1):
    curr_file = open(f"./transcription-{video_title}/process_chunks/chunk{i}.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", curr_file)

    # Salva a transcrição de cada trecho em um arquivo de texto
    with open(f"./transcription-{video_title}/videotext-{video_title.replace(' ', '-')}.txt", "a") as f:
        f.write(transcript["text"])

    # Imprime uma mensagem informando que a transcrição do trecho foi concluída
    print(f"Transcription for chunk {i} complete!")

# Imprime uma mensagem informando que a transcrição do vídeo foi concluída
print("Transcription complete!")

# Criando um resumo do vídeo usando o ChatGPT
texto = open(f"./transcription-{video_title}/videotext-{video_title.replace(' ', '-')}.txt", "r").read()

# Utiliza a API do OpenAI ChatGPT para criar um resumo do texto da transcrição, em português do Brasil
resumo = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Realize um resumo do texto da transcrição usando tópicos que deve ser escrito em português do Brasil."},
        {"role": "user", "content": texto},
        {"role": "system", "content": "Resumo:"},
    ],
    temperature=0,
    max_tokens=1024,
)

# Salva o resumo do vídeo em um arquivo de texto
with open(f"./transcription-{video_title}/resumetext-{video_title.replace(' ', '-')}.txt", "a") as f:
    f.write(resumo["choices"][0]["message"]["content"])

# Imprime uma mensagem informando que o resumo do vídeo foi criado
print("Summary complete!")

# Marca o fim do tempo de execução do código
fim = time.time()

# Imprime o tempo total de execução do código
print("Time to execute: " + str(int(fim - inicio)) + " seconds!")
