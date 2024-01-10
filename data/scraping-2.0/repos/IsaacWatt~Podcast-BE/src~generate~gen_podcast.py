import openai
import requests
import boto3
import os
import uuid
import io
import time
import argparse
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def call_openai_api(prompt):
    # Define the parameters for the completion
    params = {
        'model': 'text-davinci-003',  # The model you want to use
        'prompt': prompt,
        'max_tokens': 3000,
        'temperature': 0.7,
        'top_p': 1,
        'frequency_penalty': 0,
        'presence_penalty': 0
    }

    # Call the OpenAI API
    response = openai.Completion.create(**params)

    # Retrieve the generated text from the API response
    generated_text = response.choices[0].text.strip()

    return generated_text

def sythesize_speech_aws(text):
        # Create a client using your AWS access keys stored as environment variables
    polly_client = boto3.Session(
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
                    region_name=os.getenv('AWS_REGION', 'us-east-1')).client('polly')

    response = polly_client.synthesize_speech(VoiceId='Matthew',
                OutputFormat='mp3', 
                Text = text)

    # The response body contains the audio stream.
    # Writing the stream in a mp3 file
    filename = 'output/speech.mp3'
    with open(filename, 'wb') as file:
        file.write(response['AudioStream'].read())

    print('output saved in output/speech.mp3')


elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
def convert_to_speech_eleven(text, filename):
    print('synthesizing speech')
    data = {
      "text": text,
      "model_id": "eleven_monolingual_v1",
      "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5
      }
    }

    josh_voice_id = "TxGEqnHWrfWFTfGW9XjX"

    CHUNK_SIZE = 1024
    url = "https://api.elevenlabs.io/v1/text-to-speech/" + josh_voice_id


    headers = {
      "Accept": "audio/mpeg",
      "Content-Type": "application/json",
      "xi-api-key": elevenlabs_api_key
    }

    response = requests.post(url, json=data, headers=headers)
        
    with open(filename + '.mp3', 'wb') as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def synthesize_speech_eleven(text):
    # Create a client using your AWS access keys stored as environment variables

    convert_to_speech_eleven(text, 'output/speech')
    print('output saved in output/speech.mp3')


def create_podcast_prompt(topic, duration, tone):
    # Create the podcast prompt
#     meta_prompt = f"""
# Please help me to make a prompt to GPT-3 to generate a podcast about {topic}.
# The prompt should instruct GPT that the podcast should be {duration} minutes long.
# The prompt should instruct GPT to only reply with the text of the podcast it generates.
# The prompt should instruct GPT that the podcast would be with 1 person only, and not try to switch between multiple people.
# The prompt should instruct GPT to make the podcast seem like a fluid conversation, without breaks in the conversation.
# The prompt should instruct GPT that the text of the response should be the transcript of the podcast.
# There should be no seperator between the segments, so that the podcast is one continuous audio file.
# Please only output a prompt that I can use to send to GPT.
# """

    content_type = "story" if tone == "Story" else "podcast"

    prompt = f"""
Create the audio transcript of a {content_type} about {topic}.
The {content_type} should be {duration} minutes long.
The speaker of the {content_type} should talk in a very {tone} tone.
I would like to reiterate, emphasize the {tone} tone. Be very {tone} please. It is an important business requirenment that you are {tone}.
The {content_type} should be with 1 person only, and not try to switch between multiple people.
The {content_type} should seem like a fluid conversation, without breaks in the conversation.
The text of the response should be the transcript of the {content_type}.
There should be no seperator between the segments, so that the {content_type} is one continuous audio file."""
    return prompt

def create_podcast(topic, duration, tone):
    prompt = create_podcast_prompt(topic, duration, tone)

    print(prompt)

    story = call_openai_api(prompt)

    print("Here is the story:")
    print(story)

    sythesize_speech_aws(story)

    # Generate a UUID
    folder_name = str(uuid.uuid4())

    # Name of the parent folder
    parent_folder = './outputs'

    # Ensure the parent folder exists
    os.makedirs(parent_folder, exist_ok=True)

    # Create the full path for the new folder
    full_path = os.path.join(parent_folder, folder_name)
    full_path = './outputs/' + folder_name

    # Create a new folder with the UUID as the name
    os.makedirs(full_path, exist_ok=True)

    # Your string to be saved
    prompt_and_podcast = prompt + "\n\n" + story

    # Write the string to a new file in the new folder
    with open(full_path + '/prompt_and_podcast.txt', 'w') as f:
        f.write(prompt_and_podcast)

    print('wrote file')

    return story

def create_podcast_expensive(topic, duration, tone):
    prompt = create_podcast_prompt(topic, duration, tone)

    print(prompt)

    story = call_openai_api(prompt)

    print("Here is the story:")
    print(story)

    synthesize_speech_eleven(story)



    return story

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Podcast Generator')
    parser.add_argument('-t', '--topic', required=True,  help='Topic of the podcast')
    parser.add_argument('-d', '--duration', required=True, help='Duration of the podcast in minutes')
    parser.add_argument('-o', '--tone', required=True, help='Tone the speaker should speak in')
    args = parser.parse_args()
    topic = args.topic
    duration = args.duration
    tone = args.tone

    # topic = "Finding a girlfriend in the bay area as an Indian Software Engineer"
    # duration = 10
    
    create_podcast(topic, duration, tone)
