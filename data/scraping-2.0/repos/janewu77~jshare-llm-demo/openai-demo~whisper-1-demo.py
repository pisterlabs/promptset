import cfg.cfg
import openai
from datetime import datetime
from dateutil import rrule


# Whisper-1的例子
# https://platform.openai.com/docs/guides/speech-to-text?lang=python
def demo_whisper_1_run():
    # 音频转文字

    start_time = datetime.now()
    # audio_file = open("../resources/w1-cn.mp3", "rb")
    audio_file = open("../resources/w1-en.mp3", "rb")
    transcription = openai.Audio.transcribe("whisper-1", audio_file)
    audio_file.close()
    print(transcription.get("text"))

    seconds = rrule.rrule(freq=rrule.SECONDLY, dtstart=start_time, until=datetime.now())
    print(f"total spend: {seconds.count()} seconds")
    print('【END】.')


def demo_whisper_2_run():
    # 翻译，目前仅支持翻成英文
    # translate: cn -> english

    start_time = datetime.now()
    audio_file = open("../resources/w1-cn.mp3", "rb")
    transcript = openai.Audio.translate("whisper-1", audio_file)
    audio_file.close()
    print(transcript.get("text"))

    seconds = rrule.rrule(freq=rrule.SECONDLY, dtstart=start_time, until=datetime.now())
    print(f"total spend: {seconds.count()} seconds")
    print('【END】.')


if __name__ == '__main__':
    demo_whisper_1_run()
