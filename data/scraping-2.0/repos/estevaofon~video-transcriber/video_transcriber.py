import openai
import ffmpeg

# Cria um FilterableStream object
stream = ffmpeg.input("video.mp4").output("audio.mp3")
# Configura o FilterableStream object
stream.global_args("-loglevel", "quiet")
# Roda o FilterableStream object
stream.run()

# Set your OpenAI API key
openai.api_key = "SUA_API_KEY"
# Abre o arquivo de audio
audio_file= open("audio.mp3", "rb")
# Transcreve o audio
transcript = openai.Audio.transcribe("whisper-1", audio_file)
transcription = transcript.get("text")
print('\n'+"Transcrição".center(30, "="))
print(transcription)