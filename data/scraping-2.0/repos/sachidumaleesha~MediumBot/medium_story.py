import openai
from youtube_transcript_api import YouTubeTranscriptApi
from prompts_library import medium_prompts

# Extract video transcript
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        transcript_text = ""
        for entry in transcript:
            transcript_text += entry['text'] + " "
        
        return transcript_text.strip()

    except Exception as e:
        print(f"Error fetching video transcript: {e}")
        return None

openai.api_key = "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"

# Generate blog post
def generate_blog_post(transcript):
    prompt = medium_prompts.medium_story_prompt.format(video_script=transcript)
    response = openai.ChatCompletion.create(  # Use ChatCompletion for chat models
        model="gpt-3.5-turbo",  # Adjust the model name as needed
        messages=[{"role": "system", "content": "You are a Medium writer."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"].strip()

    # Generate the title using OpenAI API
def generate_title(blog_post):
    prompt = "Generate a suitable title for the following blog post:\n" + blog_post
    response = openai.ChatCompletion.create(  # Use ChatCompletion for chat models
        model="gpt-3.5-turbo",  # Adjust the model name as needed
        messages=[{"role": "system", "content": "You are a Medium writer."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"].strip()