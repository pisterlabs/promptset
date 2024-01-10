import openai, os
openai.api_key = os.getenv("OPENAI_API_KEY")

audio_file = open("/Users/zm/aigcData/podcast_clip.mp3", "rb")
translated_prompt="""This is a podcast discussing ChatGPT and PaLM model. 
The full name of PaLM is Pathways Language Model."""

transcript = openai.Audio.translate("whisper-1", audio_file, prompt=translated_prompt)
print(transcript['text'])
