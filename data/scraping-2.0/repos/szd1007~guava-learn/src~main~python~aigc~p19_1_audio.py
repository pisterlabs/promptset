import openai, os
openai.api_key = os.getenv("OPENAI_API_KEY")

audio_file = open("/Users/zm/aigcData/podcast_clip.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format="srt",
                                     prompt="这是一段Onboard播客的内容")
# print(transcript['text'])
print(transcript)