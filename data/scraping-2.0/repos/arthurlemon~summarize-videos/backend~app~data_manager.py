import json
import os

import openai
from dotenv import load_dotenv
from loguru import logger
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY", None)
assert openai.api_key is not None, "No OPENAI_API_KEY environment variable set"


class DataManager:
    def __init__(self, data_file="data.json"):
        self.data_file = data_file
        self.data = self.load_data()

    def load_data(self):
        if os.path.isfile(self.data_file):
            with open(self.data_file, "r") as f:
                data = json.load(f)
        else:
            data = {}
        return data

    def save_data(self):
        with open(self.data_file, "w") as f:
            json.dump(self.data, f)

    def save_video_summary(self, video_id, summary_all, summary_sections):
        self.data[video_id] = {
            "summary_all": summary_all,
            "summary_sections": summary_sections,
        }
        self.save_data()

    def get_video_summary(self, video_id, use_cache: bool = True):
        if use_cache:
            video_data = self.data.get(video_id)
            if video_data is not None:
                return video_data

        transcript = get_transcript(video_id)
        chunks = get_section_chunks(transcript)
        summary = generate_summary(chunks)
        return summary


def get_transcript(video_id: str) -> list[dict]:
    # TODO - Handle errors with video without subtitles
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "fr"])

    for section in transcript:
        section["end"] = section["start"] + section["duration"]

    # convert to minutes
    for section in transcript:
        section["start"] = round(section["start"] / 60, 4)
        section["end"] = round(section["end"] / 60, 4)

    return transcript


def get_section_chunks(
    transcript, chunk_duration: int = 5, min_chunk_duration: float = 0.5
):
    # Group sections into chunks of approximately 5 minutes
    chunks = []
    current_chunk = {
        "start": transcript[0]["start"],
        "end": transcript[0]["end"],
        "text": transcript[0]["text"],
    }
    for section in transcript[1:]:
        if section["start"] - current_chunk["start"] <= chunk_duration:
            current_chunk["end"] = section["end"]
            current_chunk["text"] += " " + section["text"]
        else:
            chunks.append(current_chunk)
            current_chunk = {
                "start": section["start"],
                "end": section["end"],
                "text": section["text"],
            }

    chunks.append(current_chunk)

    if current_chunk["end"] - current_chunk["start"] <= min_chunk_duration:
        # Merge the last chunk with the previous chunk
        previous_chunk = chunks[-1]
        previous_chunk["end"] = current_chunk["end"]
        previous_chunk["text"] += " " + current_chunk["text"]
        chunks.pop()

    return chunks


def generate_summary(chunks, strategy: str = "openai") -> dict:
    if strategy == "openai":
        for section in chunks:
            summary = create_summary_openai(section["text"])
            section["summary"] = summary
        overall_summary = create_summary_openai(
            " ".join([section["summary"] for section in chunks]),
            summary_start="In this video",
        )
    else:  # fake summarization for testing
        for section in chunks:
            section["summary"] = "summary"
        overall_summary = "fake global summary"
    logger.info(chunks)
    return {
        "summary_all": overall_summary,
        "summary_sections": [section for section in chunks],
    }


def create_summary_openai(text: str, summary_start: str = "In this section") -> str:
    prompt = [
        {
            "role": "user",
            "content": f"Summarize in English the following transcript from a youtube video in 5 sentences or less:"
            f"\n{text}\n\nStart your summary with {summary_start}",
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0]["message"]["content"]


if __name__ == "__main__":
    video_id = "kR1buRTIKhk"
    transcript = get_transcript(video_id)
    chunks = get_section_chunks(transcript, chunk_duration=5)
    summary = generate_summary(chunks)
