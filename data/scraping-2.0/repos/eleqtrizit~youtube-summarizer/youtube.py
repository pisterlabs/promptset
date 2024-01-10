import argparse
import contextlib
import os

from openai import OpenAI
from rich.console import Console
from youtube_transcript_api import YouTubeTranscriptApi

console = Console()

# Initialize parser
parser = argparse.ArgumentParser()


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def handle_query(gpt_prompt):
    print("Generating summary (writing to STDOUT and summary.txt)...")
    messages = [{"role": "user", "content": gpt_prompt}]
    for response in client.chat.completions.create(
        model = "gpt-4-1106-preview",
        temperature = 0.7,
        max_tokens = 1010,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0,
        stream = True,
        messages = messages
    ):
        reply = ''
        with contextlib.suppress(AttributeError):
            if content := response.choices[0].delta.content:
                console.print(content, style = "#FFFFFF", end = '')  # type: ignore
                reply += content

    # save to file
    with open("summary.txt", "w") as f:
        f.write(reply)

    print("Saved to summary.txt")


def get_transcript(video_id):
    print("Fetching transcript...")
    transcript_json = YouTubeTranscriptApi.get_transcript(video_id)
    return ' '.join([x['text'] for x in transcript_json])


PROMPT = """Summarize the following transcript in markdown.
Pretend you are a college student taking notes on a lecture.

Your output should use the following template:

### Summary

### Notes

### Keywords

### Media Discussed (TV, Movies, Books, etc)

### Tools Discussed

Transcript below:
"""

if __name__ == '__main__':
    # Adding optional argument
    parser.add_argument("-v", "--VideoId", help = "Youtube Video ID")

    # Read arguments from command line
    args = parser.parse_args()
    transcript = get_transcript(args.VideoId)
    prompt = PROMPT + transcript
    handle_query(prompt)
