import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_summary(text):

    print("API KEY:", os.getenv("OPENAI_API_KEY"))
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who summarizes YouTube transcripts for people wanting to save time by getting a summary instead of watching the video."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        # Handle API request errors here
        print(f"Error in API request: {e}")
        raise Exception
