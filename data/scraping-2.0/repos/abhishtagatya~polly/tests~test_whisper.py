from dotenv import load_dotenv
from __init__ import setup_project_path

import traceback
import os

from polly.client.openai import OpenAIClient
from polly.util.whisper import Whisper


def test_whisper() -> None:
    audios = [
        {
            'path': "tests/files/lincoln.mp3",
            'text': 'Remember what Lincoln said: "A drop of honey catches more flies than a gallo of gall'
        },
        {
            'path': "tests/files/sure.mp3",
            'text': """
                        Come to think it over, I don\'t entirely agree with it myself. 
                        Not everything I wrote yesterday appeals to me today. 
                        I am glad to learn what you think on the subject. 
                        The next time you are in the neighborhood you must visit us and we\'ll get this subject threshed ou for all time
                    """
        }
    ]
    whisper = Whisper(
        client=OpenAIClient(
            token=os.getenv('OPENAI_TOKEN'),
            model_dict={
                'OPENAI_WHISPER': os.getenv('OPENAI_WHISPER')
            }
        )
    )

    for audio in audios:
        filepath = audio['path']
        text = audio['text']

        print('Transcribing\t: ', os.path.basename(filepath))
        file = whisper.load_audio(filepath=filepath)
        try:
            transcription = whisper.transcribe(audio_file=file)
        except Exception as err:
            traceback.print_exc()
            continue

        print('Text\t: ', text)
        print('whisper\t: ', transcription)


if __name__ == '__main__':
    load_dotenv()
    setup_project_path()

    test_whisper()
