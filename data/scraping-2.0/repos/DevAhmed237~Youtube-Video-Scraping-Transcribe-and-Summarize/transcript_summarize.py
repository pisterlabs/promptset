import openai
import os


def summarize_transcript(video_id):
    try:
        # Set OpenAI API Key
        openai.api_key = open("OPENAI_API_KEY.txt", "r").read().strip()

        # Read transcript from file
        transcript = open(f'transcripts/{video_id}.txt', 'r', encoding='utf-8').read()

        # Summarize transcript using OpenAI's GPT-4 model
        response = openai.ChatCompletion.create(
        model='gpt-4',
        messages = [
            {'role':'system', 'content':  """Given a transcript from a YouTube video, your task is to rewrite the transcript as an article in a casual and easy way to understand for semi-technical people. Make sure that the length of the article is proportional to the length of the transcript"""},
            {'role':'user', 'content': f"Video transcript: \n{transcript}"}])
        
        response = response['choices'][0]['message']['content'].strip()

        # Create summaries directory if it doesn't exist
        if not os.path.exists('summaries'):
            os.makedirs('summaries')

        # Write summary to file
        with open(f'summaries/{video_id}.txt', 'w', encoding='utf-8') as txt_file:
            txt_file.write(response)

        return True
    except:
        return False



# video_id = '8i3yvypt1F4'
# summarize_transcript(video_id)