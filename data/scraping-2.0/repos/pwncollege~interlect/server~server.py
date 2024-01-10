import random
import datetime
import os
import pathlib
import json

import yaml
import openai
from flask import Flask, render_template, request, stream_with_context, redirect, url_for


"""
TODO: session
The user should not be able to specify the question / messages.
We should have a session which manages the state.

TODO: analytics
We should track video progress in depth:
- Seconds watched
- Pauses
- Jumps
- 2x speed
- Questions
- Answers
"""

SYSTEM_PROMPT = """
You are a professional educator helping a learner with a question or comment about a lecture video.
Act like you are right there with the learner, and you are both immersed in the lecture together: you are the lecturer.
You will be encouraging, supportive, and helpful.
It might be helpful to ask the learner leading questions to help them figure out the answer themselves.

"""

ASSESS_SYSTEM_PROMPT = f"""
{SYSTEM_PROMPT}
Your objective is to assess the learner's understanding of the topic and promote learner engagement.
Do not give the learner the answer!

When you are satisfied with their complete response, end the response with
###SUCCESS###
on a new line, at the end of the response.
Only focus on asessing one question before asserting success; don't follow on questions unless the learner is struggling.
"""

app = Flask(__name__)
data_path = pathlib.Path(f"/data")
chapters_data_path = data_path / "chapters"
analytics_data_path = data_path / "analytics"


def stream_gpt_completion(messages):
    result = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
        temperature=0.4,
        stream=True,
    )
    return stream_with_context((result["choices"][0]["delta"].get("content", "") for result in result))


@app.route('/')
def home():
    video_id = random.choice(list(chapters_data_path.iterdir())).stem
    return redirect(url_for('video', video_id=video_id))


@app.route("/<video_id>")
def video(video_id):
    session_id = os.urandom(16).hex()
    return render_template('home.html', video_id=video_id, session_id=session_id)


@app.route("/analytics/<video_id>")
def analytics_home(video_id):
    analytics = [path.read_text() for path in analytics_data_path.iterdir() if video_id in path.stem]
    result = ""
    for analytic in analytics:
        for line in analytic.splitlines():
            result += " - ".join(f"{k}: {v}" for k, v in json.loads(line).items()) + "<br>"
        result += "<hr>"
    return result


@app.route("/video-data/<video_id>")
def video_data(video_id):
    video_data_path = chapters_data_path / f"{video_id}.yml"
    video_data = yaml.safe_load(video_data_path.open())
    return video_data


@app.route("/discuss/<video_id>", methods=["POST"])
def discuss(video_id):
    when = request.get_json().get("when")
    messages = request.get_json().get("messages")

    video_data_path = chapters_data_path / f"{video_id}.yml"
    video_data = yaml.safe_load(video_data_path.open())

    transcript_segments = [
        transcript_segment
        for chapter in video_data
        for transcript_segment in chapter["transcript"]
    ]

    for transcript_segment in transcript_segments:
        if transcript_segment["start"] <= when < transcript_segment["stop"]:
            segment_id = transcript_segment["id"]
            break
    else:
        assert False

    transcript_context = " ".join(f"{segment['text']}" for segment in transcript_segments[max(segment_id-10, 0):segment_id+1])
    system_message = "\n".join([
        SYSTEM_PROMPT,
        "This is the surrounding lecture video transcript context where the learner currently is:\n",
        transcript_context,
    ])

    role_types = {
        "ai": "assistant",
        "user": "user",
        "assessment": "assistant",
    }
    messages = [
        dict(role=role_types[message["type"]], content=message["text"])
        for message in messages
    ]
    return stream_gpt_completion([dict(role="system", content=system_message), *messages])


@app.route("/assess/<video_id>", methods=["POST"])
def assess(video_id):
    chapter = request.get_json().get("chapter")
    messages = request.get_json().get("messages")

    video_data_path = chapters_data_path / f"{video_id}.yml"
    video_data = yaml.safe_load(video_data_path.open())
    chapter = video_data[chapter]

    transcript_context = " ".join(transcript_segment["text"] for transcript_segment in chapter["transcript"])
    system_message = "\n".join([
        ASSESS_SYSTEM_PROMPT,
        "This is the surrounding lecture video transcript context where the learner currently is:\n",
        transcript_context,
    ])

    role_types = {
        "ai": "assistant",
        "user": "user",
        "assessment": "assistant",
    }
    messages = [
        dict(role=role_types[message["type"]], content=message["text"])
        for message in messages
    ]
    return stream_gpt_completion([dict(role="system", content=system_message), *messages])


@app.route("/analytics/<video_id>", methods=["POST"])
def analytics(video_id):
    data = request.get_json()
    session_id = data.pop("session_id")
    data["datetime"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    analytics_data_path.mkdir(exist_ok=True)
    analytics_path = analytics_data_path / f"{video_id}-{session_id}.jsonl"
    with analytics_path.open("a") as analytics_file:
        analytics_file.write(json.dumps(data) + "\n")
    return ""