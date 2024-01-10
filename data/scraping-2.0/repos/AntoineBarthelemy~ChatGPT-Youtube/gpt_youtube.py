import os
from pytube import YouTube
from moviepy.editor import *
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import types
import re
from google.cloud import storage
import openai

# Secret area
openai.api_key = "#Secret"


def transcribe_audio_file(file_path, language_code):
    client = speech.SpeechClient()

    storage_client = storage.Client()
    bucket_name = "projetchat"
    blob_name = os.path.basename(file_path)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)

    audio = types.RecognitionAudio(uri=f"gs://{bucket_name}/{blob_name}")
    config = types.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,
        language_code=language_code,
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result()

    blob.delete()

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript

    return transcript


def interact_with_gpt(prompt, full_transcript, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Voici une transcription d'une vidéo YouTube.",
            },
            {"role": "user", "content": full_transcript},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
    )

    message = response.choices[0].message["content"].strip()
    return message


def process_video(youtube_url, questions):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    credentials_file_path = os.path.join(
        script_dir, "atomic-graph-385209-2255df6e2d73.json"
    )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file_path

    url = youtube_url

    youtubevideo = YouTube(url)

    mp3_audio_stream = (
        youtubevideo.streams.filter(only_audio=True, mime_type="audio/mp3")
        .order_by("abr")
        .first()
    )

    if mp3_audio_stream:
        audio_stream = mp3_audio_stream
    else:
        audio_stream = (
            youtubevideo.streams.filter(only_audio=True).order_by("abr").first()
        )

    output_path = "C:/Users/waren/OneDrive/Bureau/Informatique/Speech-to-text v2"

    print("Téléchargement")

    downloaded_file = audio_stream.download(output_path=output_path)

    print("Ok")

    if not mp3_audio_stream:
        input_audio = AudioFileClip(downloaded_file)
        safe_title = re.sub(r'[\\/*?:"<>|]', "", youtubevideo.title)
        output_audio_path = f"{output_path}/{safe_title}.mp3"
        input_audio.write_audiofile(output_audio_path, codec="mp3")
        os.remove(downloaded_file)
    else:
        output_audio_path = downloaded_file

    print(f"Fichier MP3 créé: {output_audio_path}")

    language_code = "fr-FR"

    print("Transcription en cours...")
    transcript = transcribe_audio_file(output_audio_path, language_code)

    gpt_responses = []
    for question in questions:
        gpt_response = interact_with_gpt(question, transcript)
        gpt_responses.append(gpt_response)

    return gpt_responses, transcript


if __name__ == "__main__":
    # Exemple d'utilisation de la fonction process_video
    url = "https://www.youtube.com/watch?v=nua_0e0epG4&ab_channel=Camille%26Justine"
    questions = [
        "Fais moi un résumé de la transcription en Anglais:",
        "Fais moi en un résumé en français:",
        "Quel est le ton général ou l'émotion principale exprimée dans la transcription suivante :",
    ]
    gpt_responses = process_video(url, questions)
    for i, response in enumerate(gpt_responses, start=1):
        print(f"Question {i}: {questions[i-1]}\n\nRéponse: {response}\n")
