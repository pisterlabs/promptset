import openai

openai.api_key = ""

if __name__ == '__main__':
    file_path = "temp-part_1_5.mp3"
    # file_path = "temp_short_en.mp3"
    # file_path = "temp-part_1_1_en.mp3"
    audio_file = open(file_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format="text")
    print(transcript)
