import os
import openai
from pydub import AudioSegment
from dotenv import load_dotenv
from gpt_3_fine_tuning.models import get_audio_transcription, get_completion

# convert m4a to wav
# ffmpeg -i audio1837788862.m4a -acodec pcm_s16le -ac 1 -ar 16000 audio1837788862.wav


def config():
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_KEY')

# Convert a file from mp4 to mp3
# ffmpeg -i "PowerFlow Standup Call-20230328_100138-Meeting Recording.mp4" -acodec libmp3lame -ac 1 -ar 16000 "PowerFlow Standup Call-20230328_100138-Meeting Recording.mp3"


def audio_to_text(filename):
    audio = AudioSegment.from_mp3(filename)
    ten_minutes = 10 * 60 * 1000
    transcript = []
    process = len(audio) // ten_minutes
    for i in range(0, len(audio), ten_minutes):
        first_10_minutes = audio[i:i + ten_minutes]
        first_10_minutes.export("C:/Users/rudae/Downloads/audio.mp3", format="mp3")
        audio_file = open("C:/Users/rudae/Downloads/audio.mp3", "rb")
        result = get_audio_transcription(audio_file)
        result_chunks = [result['text'][i:i + 2000] for i in range(0, len(result['text']), 2000)]
        generated_chunks = []
        for transcript_chunk in result_chunks:
            message = f"Beautify the following Transcript: \n\nTranscript: {transcript_chunk}\n\n Beautified Transcript: \n"
            response = get_completion(message, max_tokens=1000)

            generated_chunks.append(response)
        transcript.append("".join(generated_chunks))
        print(f"Processed {i // ten_minutes + 1} of {process + 1} chunks")
    return transcript


def main():
    """
    The main function that is called when the script is run.
    :return: None
    """
    config()
    transcript = audio_to_text("C:/Users/rudae/Downloads/Meeting on AI Project-20230329_103809-Meeting Recording.mp3")
    with open("C:/Users/rudae/Downloads/Meeting on AI Project-20230329_103809-Meeting Recording.txt", "w") as f:
        f.write(" ".join(transcript))


if __name__ == "__main__":
    main()
