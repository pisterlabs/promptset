from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from pydub import AudioSegment
from sys import argv
import re
import time
import json

# use ./key.txt as the key file
with open("./key.txt", "r") as f:
    key = f.read().strip()

# create an OpenAI object
client = OpenAI(api_key=key)


def create_transcript_from_audio_file(audio_file: str, language: str = "fr"):
    file = open(audio_file, "rb")
    response = client.audio.transcriptions.create(
        model="whisper-1", language=language, file=file, response_format="verbose_json"
    )
    new_file = open(f"output.json", "w")
    new_file.write(json.dumps(response.segments))  # type: ignore # there is no valid type for verbose json Transcription response
    new_file.close()


def create_audio_file_from_text(text: str) -> str:
    chunks = []
    # create semantically meaningful chunks
    text_split_by_punctuation = re.split(r"(?<=[.!?])\s+", text)
    temp_chunk = ""
    for i in range(0, len(text_split_by_punctuation)):
        if len(temp_chunk) + len(text_split_by_punctuation[i]) < 4000:
            temp_chunk += text_split_by_punctuation[i] + "."
        else:
            chunks.append(temp_chunk)
            temp_chunk = ""
    chunks.append(temp_chunk)

    audio_files = []
    for i, chunk in enumerate(chunks):
        print(f"Creating audio file chunk {i}/{len(chunks)}...")
        file_name = f"temp/output-temp-{i}.mp3"
        response = client.audio.speech.create(
            model="tts-1-hd", voice="alloy", input=chunk, speed=1
        )
        response.stream_to_file(file_name)
        audio_files.append(file_name)

    # combine audio files
    print("Combining audio files...")
    combined_file_name = f"output-{time.time()}.mp3"
    combined = AudioSegment.from_file(audio_files[0], format="mp3")
    for file in audio_files[1:]:
        next_segment = AudioSegment.from_file(file, format="mp3")
        combined += next_segment

    combined.export(combined_file_name, format="mp3")

    print("Audio files combined.")

    return combined_file_name


def translate_and_reformat_transcript(
    transcript: str, target_language: str
) -> str | None:
    # go by 4000 char chunks
    translated_transcript = ""
    for i in range(0, len(transcript), 4000):
        print(f"Translating chunk {i} to {i+4000}...")
        chunk = transcript[i : i + 4000]
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": f"system",
                    "content": "You are a translation and reformatting bot. You only output the requested translation in {target_language} with no further commentary.",
                },
                {
                    "role": "user",
                    "content": f"Translate the following transcript into {target_language} and reformat it with punctuation: {chunk}",
                },
            ],
        )
        if resp.choices[0].message.content is not None:
            translated_transcript += " " + resp.choices[0].message.content
    return translated_transcript


def retrieve_youtube_transcript(youtube_id: str) -> str:
    transcript_resp = YouTubeTranscriptApi.get_transcript(youtube_id)
    # turn the transcript into a string
    transcript = TextFormatter().format_transcript(transcript_resp)
    return transcript


def create_personal_podcast(youtube_id: str, iso_language_code: str = "fr"):
    print("Creating personal podcast for video: " + youtube_id)
    print("Retrieving transcript for video: " + youtube_id)
    transcript = retrieve_youtube_transcript(youtube_id)
    print("Transcript retrieved.")
    print("Translating and reformatting transcript...")
    translated_transcript = translate_and_reformat_transcript(
        transcript, iso_language_code
    )
    if translated_transcript is None:
        print("Error translating and reformatting transcript.")
        exit(1)
    print("Translation and reformatting complete.")
    print("Creating audio file...")
    audio_file = create_audio_file_from_text(translated_transcript)
    print("Audio file created.")
    # print("Transcribing audio file...")
    # create_transcript_from_audio_file(audio_file, iso_language_code)

    # print("Audio file transcribed.")


if __name__ == "__main__":
    if len(argv) < 3:
        print("Usage: python personal-podcast.py <youtube_id> <iso_language_code>")
        exit(1)
    else:
        youtube_id = argv[1]
        iso_language_code = argv[2]
        create_personal_podcast(youtube_id, iso_language_code)
