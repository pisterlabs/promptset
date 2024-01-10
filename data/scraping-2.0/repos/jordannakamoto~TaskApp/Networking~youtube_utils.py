from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import os
#import nltk
# need to use the downloder to get wordlists on first run
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import sys
from dotenv import load_dotenv


# Set your OpenAI API key
load_dotenv()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()
maximum_context_length = 4097

def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([item['text'] for item in transcript_list]).replace("\n", " ")
        return transcript
    except Exception as e:
        print("An error occurred fetching the transcript:", e)
        return None

def filter_transcript(transcript):
    stop_words = set(stopwords.words('english'))  # You can adjust the language
    word_tokens = word_tokenize(transcript)

    filtered_transcript = [word for word in word_tokens if word.lower() not in stop_words]
    filtered_transcript_length = len(filtered_transcript)  # Length of the transcript in words

    return ' '.join(filtered_transcript), filtered_transcript_length

def process_chunk(chunk, model_name="gpt-3.5-turbo"):
    prompt = f"Compose a succinct one sentence instruction based on the following content, try to use an action word other than 'create':\n{chunk}"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print("An error occurred with OpenAI:", e)
        return None

def process_transcript(transcript,length):
    # Prepare task groups
    task_groups = []
    tasks_per_group = 5

    chunks = chunk_transcript_by_tokens(transcript,length/((8*tasks_per_group)/7))
    tasks = []

    # Process each chunk and add to tasks list
    for chunk in chunks:
        task = process_chunk(chunk)
        # print(task)
        tasks.append(task)  # Add the task to the list

    

    for i in range(0, len(tasks), tasks_per_group):
        task_groups.append(tasks[i:i + tasks_per_group])

    # Generate and edit group titles
    titled_task_dict = {}

    for i, tasks_in_group in enumerate(task_groups, start=1):
        # Call generate_group_title to get a title for each group
        group_title = generate_group_title(tasks_in_group)
        titled_task_dict[group_title] = tasks_in_group

    # Print task groups with their new titles and counts
    for title, tasks_in_group in titled_task_dict.items():
        print(f"{title} has {len(tasks_in_group)} tasks:")
        for i, task in enumerate(tasks_in_group, 1):
            print(f"  {i}. {task}")
        print()  # Add an extra newline for spacing between groups

    return titled_task_dict

def chunk_transcript_by_tokens(transcript, approx_max_tokens):
    """
    Chunk the transcript into approximate token-sized chunks.
    This is a heuristic approach and may not perfectly align with the actual token count.
    """
    words = transcript.split()
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # Adding 1 for space or tokenization characters

        if current_length >= approx_max_tokens:
            yield ' '.join(current_chunk)
            current_chunk = []
            current_length = 0

    if current_chunk:
        yield ' '.join(current_chunk)

def generate_group_title(tasks):
    # Combine the tasks into a single string for the prompt
    tasks_combined = " ".join(tasks)
    
    # Create the prompt for the AI model
    prompt = (f"I have a set of subtasks related to a single item in a to-do app. "
              f"Here are the tasks: {tasks_combined} "
              f"What would be a brief but descriptive 6 word maximum title for this group of tasks?")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "user", "content": prompt}
                ],
            max_tokens=64,  # Adjust as needed
            temperature=0.7  # Slightly creative but focused
        )

        # Extract and print the title
        title = response.choices[0].message.content
        return title
    except Exception as e:
        print(f"An error occurred while generating the title: {e}")
        return None

# For smaller video size
def summarize_tasks(tasks):
    prompt = f"Can you summarize this into 8 main tasks and 3 subtasks for each to paste into my todo app user interface: {tasks}"
    if prompt:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            print(completion.choices[0].message.content)
            return None
        except Exception as e:
            print("An error occurred with OpenAI:", e)
            return None
    else:
        return "No prompt generated."

def summarize_video(video_id):
    prompt = generate_prompt_for_video(video_id)
    if prompt:
        try:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                stop=["\n"],
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            print("An error occurred with OpenAI:", e)
            return None
    else:
        return "No prompt generated."

def parse_video_id(url):
    if 'youtube.com' in url:
        # Extract video_id from regular YouTube URL
        query_string = url.split('?')[-1]
        parameters = query_string.split('&')
        for param in parameters:
            if '=' in param:  # Check if the parameter contains '='
                key, value = param.split('=')
                if key == 'v':
                    return value.split('&')[0]  # Return only the video ID part
    elif 'youtu.be' in url:
        # Extract video_id from shortened YouTube URL
        video_id = url.split('/')[-1]
        return video_id.split('?')[0]  # Ignore anything after '?'
    return None  # Return None if the URL format is not recognized

# Example Usage

if len(sys.argv) > 1:
    youtube_url = sys.argv[1]  # Get the YouTube URL from command-line argument
else:
    youtube_url = "default_url"  # Or some default value

video_id = parse_video_id(youtube_url)
transcript = get_transcript(video_id)
transcript, length = filter_transcript(transcript)
print("Length in words:", length)
process_transcript(transcript,length)
