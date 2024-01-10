import openai
from openai import OpenAI
from my_api_secrets import get_api_key

client = OpenAI(api_key=get_api_key())


def speech2text(output_lan="en"):
    # Generates audio from the input text.
    audio_file = open("../data/kenan.mp3", "rb")  # 之后改为语音输入接口
    if output_lan == "en":
        transcript = client.audio.translations.create(
            model="whisper-1",
            file=audio_file,  # in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
            response_format="text",  # The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
            temperature=0,  # defaults to 0

        )
    else:
        # 返回音频自身的语言
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text",
        )

    print(transcript)
    return transcript


def text2speech(input_str="Hello world! This is a streaming test."):
    # Generates audio from the input text.
    # Transcribes audio into the input language.
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",  # The voice to use when generating the audio. Supported voices are alloy, echo, fable, onyx, nova, and shimmer
        input=input_str,  # 改为LLMs输出
        speed=1,  # The speed of the generated audio
    )

    return response.stream_to_file("../data/output.mp3")
