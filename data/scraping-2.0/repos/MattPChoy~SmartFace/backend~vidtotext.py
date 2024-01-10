# from keys import open_ai_key
import openai

def transcribe(path):
    """
    REMEBER TO ADD YOUR SECRET KEY TO CONDA ENVIRONMENT VARIABLES

    Transcribe a video file into text.
    :param path: Path to the video file.
    :return: String representing the text of the video.
    """

    # Load the video file
    audio_file= open(path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript


# if __name__ == "__main__":
#     print(transcribe("./videos/videoplayback.mp4"))




