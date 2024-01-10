from openai import OpenAI
from pathlib import Path
import os
import sys
import json
import dotenv
import slugify
import requests
import uuid
import pathlib
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip

# Load environment variables
dotenv.load_dotenv()

# set up the openai client

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# prompt the user for the topic
topic = input("What topic should the AI debate?")
proponent_name = input("What is the name of the proponent?")
opponent_name = input("What is the name of the opponent?")

moderator_model = "gpt-3.5-turbo-1106"
#proponent_model = "gpt-3.5-turbo-1106"
proponent_model = "gpt-4-1106-preview"
#opponent_model = "gpt-3.5-turbo-1106"
opponent_model = "gpt-4-1106-preview"
image_description_model = "gpt-3.5-turbo-1106"


# initialize the list of messages
debate_messages = []

# create a unique name for the debate
topic_slug = slugify.slugify(topic)
debate_id = str(uuid.uuid4())
debate_folder = f"debates/{topic_slug}-{debate_id}"

# create the debates folder if it doesn't exist
if not os.path.exists("debates"):
    os.makedirs("debates")

# create the debate folder if it doesn't exist
if not os.path.exists(f"{debate_folder}"):
    os.makedirs(f"{debate_folder}")


def get_moderator_response(debate_messages, instruction=""):
    response = client.chat.completions.create(
        model=moderator_model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"You are the moderator. You are moderating the debate between {proponent_name} and {opponent_name} on {topic}. Your response should be a json object with the keys 'speaker' and 'content'",
            },
            {
                "role": "user",
                "content": json.dumps(debate_messages),
            },
            {
                "role": "user",
                "content": instruction,
            },
        ],
    )

    response = json.loads(response.choices[0].message.content)

    debate_messages.append(response)

    return response


def get_proponent_response(debate_messages):
    response = client.chat.completions.create(
        model=proponent_model,
        response_format={"type": "json_object"},
        temperature=1.1,
        messages=[
            {
                "role": "system",
                "content": f"You are {proponent_name}. You are debating as a proponent of {topic}. Your response should be a json object with the keys 'speaker' and 'content'. Close out the video by thanking the viewers for watching, asking them to like the video and subscribe to our channel.",
            },
            {
                "role": "user",
                "content": json.dumps(debate_messages),
            },
        ],
    )
    response = json.loads(response.choices[0].message.content)
    debate_messages.append(response)
    return response


def get_opponent_response(debate_messages):
    response = client.chat.completions.create(
        model=opponent_model,
        response_format={"type": "json_object"},
        temperature=1.1,
        messages=[
            {
                "role": "system",
                "content": f"You are {opponent_name}. You are debating as an opponent of {topic}. Your response should be a json object with the keys 'speaker' and 'content'",
            },
            {
                "role": "user",
                "content": json.dumps(debate_messages),
            },
        ],
    )

    response = json.loads(response.choices[0].message.content)

    debate_messages.append(response)

    return response


def debate_round(debate_messages, moderator_instruction=""):
    moderator_response = get_moderator_response(
        debate_messages, instruction=moderator_instruction
    )
    proponent_response = get_proponent_response(debate_messages)
    opponent_response = get_opponent_response(debate_messages)
    return debate_messages


# Moderator starts the debate and asks for opening statements
debate_messages = debate_round(
    debate_messages,
    moderator_instruction=f"Introduce the debate on {topic}. Ask {proponent_name} and {opponent_name} for their constructive statements. Each side will present their constructive statement before you get a chance to speak again.",
)

# Moderator facilitates cross-examination
debate_messages = debate_round(
    debate_messages,
    moderator_instruction=f"Start the rebuttal round. Each side will present their rebuttal before you get a chance to speak again.",
)

# Moderator gives closing remarks
response = get_moderator_response(
    debate_messages,
    instruction=f"Moderator, please give your closing remarks and formally conclude the debate on {topic}.",
)

# print the debate messages
print(json.dumps(debate_messages, indent=1))

# write the transcript to a file in the debate folder
with open(f"{debate_folder}/transcript.json", "w") as f:
    f.write(json.dumps(debate_messages, indent=1))

for i, message in enumerate(debate_messages):
    # valid voice names are alloy, echo, fable, onyx, nova, and shimmer
    if message["speaker"] == "moderator":
        voice = "alloy"
    elif message["speaker"] == proponent_name:
        voice = "echo"
    elif message["speaker"] == opponent_name:
        voice = "fable"
    else:
        voice = "alloy"
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=message["content"],
    )
    # save the audio to a file in the debate folder
    speech_file_path = Path(f"{debate_folder}/{i}.mp3")
    response.stream_to_file(speech_file_path)

# iterate through the conversation and create an image for each message.

for i, message in enumerate(debate_messages):
    text_response = client.chat.completions.create(
        model=image_description_model,
        messages=[
            {
                "role": "system",
                "content": "You will be provided with some text. Your response should be a description of an image collage to accompany the text. The collage will be displayed on screen while the text is being read aloud. Avoid anything explicit or inappropriate. Avoid anything offensive. Avoid anything that is copyrighted. If copyrighted works are mentioned, use generic descriptions instead.",
            },
            {"role": "user", "content": message["content"]},
        ],
    )
    visual_description = text_response.choices[0].message.content
    print(visual_description)
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"{visual_description}",
            size="1024x1024",
            quality="hd",
            style="vivid",
            n=1,
        )
    except:
        print("Image generation failed.")
        continue
    image_url = response.data[0].url
    image = requests.get(image_url)
    with open(f"{debate_folder}/{i}.jpg", "wb") as f:
        f.write(image.content)

# start creating a video from the images and audio files


directory = debate_folder

# get all jpg files in the directory

jpg_files = list(pathlib.Path(directory).glob("*.jpg"))

# get all mp3 files in the directory

mp3_files = list(pathlib.Path(directory).glob("*.mp3"))

print(f"{mp3_files}\n\n{jpg_files}")

# sort the files by name

jpg_files.sort()
mp3_files.sort()

# create a list of clips

clips = []

# for example, 0.jpg and 0.mp3 should be combined into a video clip.
# the video clip should match the duration of the mp3 file.

for i in range(len(jpg_files)):
    jpg_file = str(jpg_files[i])
    mp3_file = str(mp3_files[i])
    print(f"Processing {jpg_file} and {mp3_file}")
    jpg_clip = ImageClip(jpg_file)
    mp3_clip = AudioFileClip(mp3_file)
    jpg_clip = jpg_clip.set_duration(mp3_clip.duration)
    jpg_clip = jpg_clip.set_audio(
        mp3_clip
    )  # Set the audio of the jpg_clip to be the mp3_clip
    clips.append(jpg_clip)

# concatenate the clips
final_clip = concatenate_videoclips(clips)

# write the final clip to a file

final_clip.write_videofile(f"{directory}/final_clip.mp4", fps=24)

# close the final clip

final_clip.close()
