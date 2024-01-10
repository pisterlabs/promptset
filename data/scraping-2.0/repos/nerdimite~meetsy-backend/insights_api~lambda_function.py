import os
import json
import openai
from utils import format_transcript_str, postprocess_points

openai.api_key = os.getenv("OPENAI_API_KEY")

def invoke_gpt3(prompt, max_tokens=512, temperature=0.5):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()
    
def generate_minutes_of_meeting(transcript_str):

    mom_prompt = f"""Generate a meeting summary/notes for the following transcript:
    Meeting Transcription:
    {transcript_str}

    Instructions:
    1. Do not use the same words as in the transcript.
    2. Use proper grammar and punctuation.
    3. Use bullets to list the points.
    4. Add as much detail as possible.

    Meeting Minutes:
    -"""

    raw_minutes = invoke_gpt3(mom_prompt, temperature=0.5)
    minutes = postprocess_points(raw_minutes)

    return minutes
    
def generate_action_items(transcript_str):

    action_prompt = f"""Extract the Action Items / To-Do List from the Transcript.
    Meeting Transcription:
    {transcript_str}

    Action Items:
    -"""
    raw_action_items = invoke_gpt3(action_prompt, temperature=0.4)
    action_items = postprocess_points(raw_action_items)

    return action_items


def lambda_handler(event, context):
    
    body = json.loads(event.get('body')) if type(event.get('body')) == str else event.get('body')
    
    transcript_str = format_transcript_str(body['transcript'])
    minutes = generate_minutes_of_meeting(transcript_str)
    action_items = generate_action_items(transcript_str)
    
    output = {
        "minutes": minutes,
        "action_items": action_items
    }
    
    return {
        'statusCode': 200,
        'body': output,
        'headers': {
            'Cache-Control': 'no-cache, no-store', 
            'Content-Type': 'application/json',
        }
    }
