import openai # openai api
from youtube_transcript_api import YouTubeTranscriptApi # getting transcript from youtube video
from dotenv import load_dotenv # loading api keys from enviroment
import os # loading api keys from enviroment

load_dotenv() # loading api keys from enviroment
open_api_key = os.environ.get("OPENAI_API_KEY")

def get_youtube_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([caption['text'] for caption in transcript])

    return transcript_text


def summarize_text(text):
    # Set up OpenAI API credentials
    openai.api_key = open_api_key  # Replace with your OpenAI API key

    # Define the chat completion prompt
    prompt = f"start as 'This video is about' and summarize the following in no more than 60 words: {text}."

    # Generate the response using OpenAI's ChatGPT model
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=150,  # Adjust the desired length of the summary
        temperature=0.3,  # Adjust the level of randomness in the response
        n=1,
        stop=None,
        timeout=15  # Adjust the timeout as needed
    )

    # Extract the summarized text from the API response
    summary = response.choices[0].text.strip()
    print(response)
    return summary

# Test the script with a YouTube video ID
video_id = "Qa4K7XsRO0g"  # Replace with the ID of the YouTube video

# Get the transcript of the YouTube video
transcript = get_youtube_transcript(video_id)
# print(transcript)

# Summarize the transcript
summary = summarize_text(transcript)

print("Summarized Text:")
print(summary)
