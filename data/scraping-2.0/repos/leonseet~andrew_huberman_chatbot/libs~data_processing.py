from dotenv import load_dotenv
from typing import List
import numpy as np
import re
from youtube_transcript_api import YouTubeTranscriptApi

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field

from llama_index.llms import OpenAI

load_dotenv()


def convert_timestamps_to_intervals(timestamps):
    """
    Convert a list of timestamps and descriptions into a list of intervals.

    Args:
        timestamps (list): A list of strings in the format "HH:MM:SS description".

    Returns:
        list: A list of dictionaries, where each dictionary represents an interval
        and has the keys "start", "end", and "desc". The "start" and "end" values
        are in seconds, and the "desc" value is a string describing the interval.
    """

    def convert_to_seconds(timestamp):
        num_splits = len(timestamp.split(":"))
        if num_splits == 2:
            timestamp = re.search(r"\d{1,2}:\d{2}", timestamp).group(0)
            m, s = map(int, timestamp.split(":"))
            h = 0
        elif num_splits == 3:
            timestamp = re.search(r"\d{1,2}:\d{2}:\d{2}", timestamp).group(0)
            h, m, s = map(int, timestamp.split(":"))
        else:
            try:
                h, m, s = map(int, timestamp.split(":")[-3:])
            except:
                timestamp = re.search(r"\d{2}:\d{2}:\d{2}", timestamp).group(0)
                h, m, s = map(int, timestamp.split(":"))
        return h * 3600 + m * 60 + s

    def split_timestamps(description):
        # Find all timestamps in the description
        timestamps = re.findall(r"\d{2}:\d{2}:\d{2}", description)

        # Split the description at each timestamp
        parts = re.split(r"\d{2}:\d{2}:\d{2}", description)
        parts = [part.strip() for part in parts if part.strip()]

        return list(zip(timestamps, parts))

    intervals = []
    for i, entry in enumerate(timestamps):
        parts = entry.split(" ", 1)
        initial_timestamp = parts[0]
        description = parts[1]

        # Handle multiple timestamps within a single description
        if len(re.findall(r"\d{2}:\d{2}:\d{2}", description)) > 0:
            split_entries = split_timestamps(description)
            for j, (ts, desc) in enumerate(split_entries):
                start_seconds = convert_to_seconds(ts)
                end_seconds = (
                    convert_to_seconds(timestamps[i + 1].split(" ")[0])
                    if i + 1 < len(timestamps)
                    else None
                )
                intervals.append(
                    {
                        "start": float(start_seconds),
                        "end": float(end_seconds) if end_seconds else "end",
                        "desc": desc,
                    }
                )
        else:
            start_seconds = convert_to_seconds(initial_timestamp)
            end_seconds = (
                convert_to_seconds(timestamps[i + 1].split(" ")[0])
                if i + 1 < len(timestamps)
                else None
            )
            intervals.append(
                {
                    "start": float(start_seconds),
                    "end": float(end_seconds) if end_seconds else "end",
                    "desc": description.strip(),
                }
            )

    intervals = [interval for interval in intervals if interval["end"] != "end"]
    intervals = [
        interval for interval in intervals if "sponsor" not in interval["desc"].lower()
    ]

    return intervals


def convert_transcript_to_dict(url):
    """
    Convert YouTube video transcript to a dictionary.

    Args:
        url (str): The URL of the YouTube video.

    Returns:
        dict: A dictionary where the keys are the timestamps of each transcript
        entry and the values are the corresponding transcript text.
    """
    match = re.search(r"v=([^&]+)", url)
    video_id = match.group(1) if match else url.split("/")[-1]
    # video_id = url.split("/")[-1]
    data = YouTubeTranscriptApi.get_transcript(video_id, languages=["en-US", "en"])

    result = {}
    for entry in data:
        if "text" in entry:
            result[entry["start"]] = entry["text"]
    return result


def extract_transcript_from_timeframe(data_dict, start, end):
    """
    Extracts the transcript of a given timeframe from a dictionary of YouTube video transcripts.

    Args:
    data_dict (dict): A dictionary containing the YouTube video transcript data.
    start (float): The start time of the desired transcript in seconds.
    end (float): The end time of the desired transcript in seconds.

    Returns:
    str: The transcript of the given timeframe.
    """

    def clean_test(text):
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        text = text.replace(" -", "-")
        text = text.strip()
        return text

    keys = np.array(list(data_dict.keys()))
    closest_start = np.argmin(np.abs(keys - start))
    closest_end = np.argmin(np.abs(keys - end))
    timestamps = keys[closest_start : closest_end + 1]

    transcript = ""
    for t in timestamps:
        text = clean_test(data_dict[t])
        transcript += text + " "

    transcript = transcript.strip()

    return transcript


def generate_timestamps_from_yt_transcript(yt_transcript):
    """
    Generate timestamps from a YouTube transcript.

    Args:
        yt_transcript (dict): The YouTube transcript.

    Returns:
        List[dict]: A list of dictionaries containing the start time, end time, description, and transcript for each timestamp.
    """

    class Transcript(BaseModel):
        """Data model for transcript."""

        start: float = Field(description="Start time of the timestamp in seconds")
        end: float = Field(description="End time of the timestamp in seconds")
        desc: str = Field(description="Short description of the timestamp")
        # transcript: str

    class Transcripts(BaseModel):
        """Data model for transcripts."""

        transcripts: List[Transcript]

    MODEL_NAME = "gpt-4-1106-preview"
    TEMPERATURE = 0.1
    GENERATION_RETRIES = 3
    PROMPT_TEMPLATE_STR = """\
    Given the Youtube Transcript below, read it till the very end, pick up the main points and create timestamps.
    All transcript time should be covered.
    Each timestamp should be more than 30 seconds long.

    Youtube Transcript:
    {yt_transcript}

    {format_instructions}
    """

    model = OpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

    parser = PydanticOutputParser(pydantic_object=Transcripts)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE_STR,
        input_variables=["yt_transcript"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    _input = prompt.format_prompt(yt_transcript=str(yt_transcript))

    # Retry if there is an error
    for _ in range(GENERATION_RETRIES):
        try:
            output = model.complete(_input.to_string())
            output = output.text
            json_output = parser.parse(output)
            str_err = None
        except Exception as e:
            str_err = str(e)
            print(str_err)
            pass

        if not str_err:
            break

    transcripts = []

    for timestamp in json_output.transcripts:
        start = timestamp.start
        end = timestamp.end
        desc = timestamp.desc
        transcript = extract_transcript_from_timeframe(
            data_dict=yt_transcript, start=start, end=end
        )
        transcript = transcript.strip()
        interval = {"start": start, "end": end, "desc": desc, "transcript": transcript}
        transcripts.append(interval)

    return transcripts
