import openai

def whisper(api_key, video_file):

    openai.api_key = api_key
    video_file = video_file
    print("\n\nReading the video....")
    response = openai.Audio.transcribe("whisper-1", video_file)

    return response.text