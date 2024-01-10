import sys
import pathlib
import yaml

import openai


SYSTEM_PROMPT = """
Extract multiple video chapters from the above lecture video transcript.
Each segment begins with an id followed by a period.
The chapters should cover the entire video, and each segment should be included in exactly one chapter.

In each video chapter, include notes, which summarize and enumerate the important points in the segment.
Also include some engaging open-ended questions that you might ask the learner to check their understanding of the segment.
These questions will be used to generate a discussion between the learner and the educator.

The video chapter format should be yml:

- title: <CHAPTER_TITLE>
  start_id: <CHAPTER_START_ID>
  notes:
    - <CHAPTER_NOTE_1>
    - <...>
  questions:
    - <CHAPTER_QUESTION_1>
    - <...>
"""

def parse_transcript(transcript_path):
    vtt = transcript_path.read_text()
    segments = vtt.split("\n\n")[1:]
    result = []
    for segment in segments:
        times, text = segment.split("\n", 1)
        start, stop = times.split("-->")
        def ms(time):
            minutes, seconds = time.split(":")
            return int(int(minutes) * 60 * 1000 + float(seconds) * 1000)
        result.append(dict(start=ms(start), stop=ms(stop), text=text))
    return result

path = pathlib.Path(sys.argv[1])
transcript_segments = parse_transcript(path)
transcript_message = "".join(f"{i}. {segment['text']}\n" for i, segment in enumerate(parse_transcript(path)))

messages = [
    dict(role="system", content=SYSTEM_PROMPT),
    dict(role="user", content=transcript_message),
]
result = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=messages,
    temperature=0.4,
)
raw_chapters = result["choices"][0]["message"]["content"]
chapters = yaml.safe_load(raw_chapters)

for chapter, next_chapter in zip(chapters, chapters[1:]):
    chapter["stop_id"] = next_chapter["start_id"] - 1
next_chapter["stop_id"] = len(transcript_segments)

for chapter in chapters:
    start_id = chapter.pop("start_id")
    stop_id = chapter.pop("stop_id")
    chapter["transcript"] = [
        dict(id=id, **segment)
        for id, segment in enumerate(transcript_segments[start_id:stop_id+1], start=start_id)
    ]

yaml.safe_dump(chapters, sys.stdout, sort_keys=False)
