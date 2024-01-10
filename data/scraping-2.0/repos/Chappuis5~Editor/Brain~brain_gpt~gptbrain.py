from Audio.audio_transcriber.audio_transcriber import Audio
import openai
import nltk


def split_text_gpt(text: str, max_tokens_per_part: int) -> list:
    """
    Splits a given text into multiple parts based on a token limit.

    :param text: The input text to be split.
    :type text: str
    :param max_tokens_per_part: The maximum number of tokens per part.
    :type max_tokens_per_part: int
    :return: A list of text parts.
    :rtype: list
    """
    # Divide the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Group sentences into parts based on the token limit
    parts = []
    current_part = ""
    for sentence in sentences:
        # If adding the next sentence doesn't exceed the token limit, add it to the current part
        if len((current_part + " " + sentence).split()) <= max_tokens_per_part:
            current_part += " " + sentence
        else:
            # Otherwise, save the current part and start a new one
            parts.append(current_part.strip())
            current_part = sentence

    # Add the last part if it's not empty
    if current_part.strip():
        parts.append(current_part.strip())

    return parts


def generate_keywords_gpt(part: str) -> list:
    """
    Generates keywords for a given part of text using GPT-4.

    :param part: Part of the text to generate keywords for.
    :type part: str
    :return: A list of keywords.
    :rtype: list
    """

    # Messages structure
    messages = [
        {"role": "system", "content": "You are an expert video editor"},
        {"role": "user", "content": f"Summarize this text: '{part}' into 10 relevant keywords for "
                                    f"video scraping, i.e., not too specific. Never any complicated words, "
                                    f"or compound words. It's imperative that the word is on-topic, "
                                    f"but absolutely never too scientific/too specific/proper noun..."
                                    f"E.G. If the text talks about ABC NEWS, "
                                    f"just output 'News', or 'TV', nothing too specific. "
                                    f"Only provide the words, in English, "
                                    f"and separate them by commas."}
    ]

    # Ask gpt-4 using the chat endpoint
    response = openai.ChatCompletion.create(model="gpt-4", messages=messages, max_tokens=350)

    keywords = [k.strip() for k in response.choices[0].message['content'].split(',')]

    return keywords[:10]


def Helper(options: dict, logger, open_ai_key: str) -> list:
    """
    Main helper function to split a script into parts and generate keywords for each part.

    :param options: Dictionary containing various options like 'script_path' and 'audio_path'.
    :type options: dict
    :param logger: A logger instance to record the process.
    :type logger: object
    :param open_ai_key: The API key for OpenAI GPT.
    :type open_ai_key: str
    :return: A list of dictionaries containing the parts, keywords, start time, and end time.
    :rtype: list
    """
    nltk.download('punkt')
    openai.api_key = open_ai_key

    # Read the content of the video script file
    with open(options["script_path"], "r") as script_file:
        video_script = script_file.read()

    # Audio transcription
    audioReader = Audio(options["audio_path"])
    logger.write('Starting audio transcription, please wait...', 'info')
    audioReader.convert_m4a_to_mp3()
    audioReader.transcribe()
    logger.write('Audio transcription completed.', 'info')

    # Read the transcriptions from the audio
    transcriptions = audioReader.get_transcriptions

    # Calculate the average reading speed based on transcriptions
    total_duration = sum(transcription["end_time"] - transcription["start_time"] for transcription in transcriptions)
    average_reading_speed = len(video_script.split()) / total_duration

    # Maximum number of words in each part based on average reading speed and a 20-second limit
    max_words_per_part = int(average_reading_speed * 10)

    # Use GPT to split the text into parts
    parts = split_text_gpt(video_script, max_words_per_part)

    # Generate keywords for each part and calculate the start and end times for each part
    parts_keywords_times = []
    total_parts = len(parts)
    part_transcriptions = []
    transcription_index = 0

    for i, part in enumerate(parts):
        # Generate keywords for each part
        keywords = generate_keywords_gpt(part)

        part_transcriptions = []
        part_word_count = 0
        while transcription_index < len(transcriptions) and part_word_count < len(part.split()):
            part_transcriptions.append(transcriptions[transcription_index])
            part_word_count += len(transcriptions[transcription_index]['word'].split())
            transcription_index += 1

        # Calculate the start and end times for the part
        start_time = part_transcriptions[0]['start_time']
        end_time = part_transcriptions[-1]['end_time'] + part_transcriptions[-1]['pause_after']

        # Store the start and end times for the part
        part_dict = {"part": part, "keywords": keywords, "start_time": start_time, "end_time": end_time}
        parts_keywords_times.append(part_dict)

        # Reset the part transcriptions for the next part
        part_transcriptions = []

    return parts_keywords_times
