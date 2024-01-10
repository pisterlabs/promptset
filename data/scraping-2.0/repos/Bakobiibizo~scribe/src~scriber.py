import datetime
import os

import openai
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
from pydub import silence


def get_audio_file():
    """

    Returns the path of the first audio file found in the "in" directory.

    The method searches for audio files with the following extensions: ".mp3", ".mp4", ".mkv", ".wav".

    If the "in" directory does not exist or there are no audio files found, it returns None.

    Returns:
        str: The path of the first audio file found in the "in" directory.

    """
    dir_path = "in"
    suffixes = (".mp3", ".mp4", ".mkv", ".wav")
    if not os.path.isdir(dir_path):
        print(f"Directory {dir_path} does not exist.")
        return
    for file in os.listdir("in"):
        if file is not None and file.endswith(suffixes):
            return f"{dir_path}/{file}"


def transcribe_audio():
    """
    Transcribes an audio file into text using OpenAI's Whisper-1 model.

    Returns:
        str: The transcribed text from the audio file.

    Raises:
        FileNotFoundError: If the audio file specified by `audio_file_name` does not exist.
    """
    audio_file_name = get_audio_file()
    try:
        with open(audio_file_name, "rb") as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file)

            with open("out/transcript.txt", "w") as file:
                file.write(transcription["text"])

        return transcription["text"]

    except FileNotFoundError:
        print(f"File {audio_file_name} not found.")
        return None


def extract_info(instruction, transcription):
    """
    Generates a response using OpenAI's ChatCompletion API based on the given instruction and transcription.

    Parameters:
    - instruction (str): The instruction given to the ChatCompletion API.
    - transcription (str): The transcription provided by the user.

    Returns:
    - str: The generated response from the ChatCompletion API.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": instruction,
            },
            {"role": "user", "content": transcription},
        ],
    )
    return response["choices"][0]["message"]["content"]


def meeting_minutes(transcription):
    """
    Generates a meeting minutes document based on a transcription.
    :param transcription: The text transcription of the meeting.
    :type transcription: str
    :return: A dictionary containing the meeting minutes information.
    :rtype: dict
    """
    roles_and_instructions = [
        (
            "abstract_summary_extraction",
            "\nYou are a highly skilled AI trained in language comprehension and summarization. I would like you to "
            "read the following text and summarize it into a concise abstract paragraph. Aim to retain the most "
            "important points, providing a coherent and readable summary that could help a person understand the "
            "main points of the discussion without needing to read the entire text. Please avoid unnecessary details "
            "or tangential points.\n",
        ),
        (
            "key_points_extraction",
            "\nYou are a proficient AI with a specialty in distilling information into key points. Based on the "
            "following text, identify and list the main points that were discussed or brought up. These should be "
            "the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your "
            "goal is to provide a list that someone could read to quickly understand what was talked about.\n",
        ),
        (
            "action_item_extraction",
            "\nYou are an AI expert in analyzing conversations and extracting action items. Please review the text "
            "and "
            "identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. "
            "These could be tasks assigned to specific individuals, or general actions that the group has decided to "
            "take. Please list these action items clearly and concisely.\n",
        ),
        (
            "sentiment_analysis",
            "\nAs an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the "
            "following text. Please consider the overall tone of the discussion, the emotion conveyed by the "
            "language used, and the context in which words and phrases are used. Indicate whether the sentiment is "
            "generally positive, negative, or neutral, and provide brief explanations for your analysis where "
            "possible.\n",
        ),
    ]
    audio_file_name = get_audio_file()
    minutes = {
        "filename": audio_file_name,
        "datetime": f"""{datetime.datetime.now().isoformat(timespec='seconds')}\n\n""",
    }
    for role, instruction in roles_and_instructions:
        minutes[role] = extract_info(instruction, transcription)

    return minutes


def scriber():
    """
    Generate a transcription and save the meeting minutes as a file.

    This function takes no parameters.

    Returns:
        None

    Raises:
        OSError: If there is an error accessing the input directory.
    """
    in_dir = "in"
    in_dir = os.path.abspath(in_dir)
    movie_suffix = [".mp4", ".mkv"]
    audio_suffix = [".mp3", ".wav"]
    audio_file_path = get_audio_file()

    for file in os.listdir(in_dir):
        if any(file.endswith(suffix) for suffix in movie_suffix):
            video = AudioFileClip(os.path.join(in_dir, file))
            audio_file_path = os.path.join(in_dir, f"{os.path.splitext(file)[0]}.mp3")
            video.write_audiofile(audio_file_path)
        elif any(file.endswith(suffix) for suffix in audio_suffix):
            audio_file_path = os.path.join(in_dir, file)

        if audio_file_path:
            segments = segment_audio()

            for i, segment in enumerate(segments):
                temp_file_name = f"out/temp_{i}.mp3"
                segment.export(temp_file_name, format="mp3")

                transcription = transcribe_audio()
                if not transcription:
                    print(f"No transcription available for {temp_file_name}.")
                    continue

                minutes = meeting_minutes(transcription)

                print(minutes)

                save_as_file(minutes)
                pretty_minutes()

                os.remove(temp_file_name)


def segment_audio():
    """
    Generates segments of audio based on detected silence.

    Returns:
        A list of audio segments.
    """
    audio = None
    audio_file_name = get_audio_file()
    if audio_file_name.endswith(".mp3"):
        audio = AudioSegment.from_mp3(audio_file_name)
    if audio_file_name.endswith(".wav"):
        audio = AudioSegment.from_wav(audio_file_name)
    detected_silence = silence.detect_nonsilent(
        audio, min_silence_len=500, silence_thresh=-32
    )
    return [audio[start:end] for start, end in detected_silence]


def pretty_minutes():
    """
    Reads the content of the "transcript.txt" file and splits it into sentences.
    Each sentence is then cleaned and written to the "full_transcript.txt" file.

    Returns:
        The number of cleaned sentences that were written to the "full_transcript.txt" file.
    """
    with open("out/transcript.txt", "r") as file:
        raw = file.read().split(".", -1)
        for each in raw:
            cleaned = f"{each.strip()}\n"

            with open("out/full_transcript.txt", "a") as f:
                return f.write(cleaned)


def save_as_file(minutes):
    """
    Save the given minutes as a human-readable text file.

    Parameters:
    - minutes (dict): A dictionary containing the minutes to be saved.

    Returns:
    - None
    """
    human_readable_text = "".join(
        f"{key}: {value}\n" for key, value in minutes.items()
    )
    with open("out/minutes.txt", "w") as file:
        file.write(human_readable_text)


if __name__ == "__main__":
    scriber()
