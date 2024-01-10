import os
import uuid
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import openai

# from transformers import pipeline, set_seed

## Setup Text Generation Pipeline
# generator = pipeline('text-generation', model='gpt2')

## Setup Random State for reducing randomness & enhance reproducibility
# set_seed(42)

openai.api_key = os.getenv('OPENAI_API_KEY')

model_name = "gpt-3.5-turbo"

rules = """
    You are a chatbot that summarizes texts. 
    You are working with transcripts from a podcast.
    You need to summarize text to no more than 200-300 words.
    Your summaries should be coherent and grammatically correct.
    You should not change the meaning of the text.
    You should not add any new information.
    You should write by points.
    Summarize the following text:
"""


def get_summary(prompt):
  try:
    response = openai.ChatCompletion.create(model=model_name,
                                            messages=[{
                                              'role': 'system',
                                              'content': rules
                                            }, {
                                              'role': 'user',
                                              'content': prompt
                                            }])

    chatbot_response = response.choices[0].message['content'].strip()
    return chatbot_response
  except Exception as e:
    print("Error occured while generating summary: ", e)
    return "Sorry, but something went wrong. Please try again later."


def download_audio_from_youtube(video_link):
  destination = 'audio/'

  try:
    video = YouTube(video_link)
    audio = video.streams.filter(only_audio=True).first()
    audio.download(output_path=destination)

    default_filename = audio.default_filename
    filename = video.title
    new_filename = f"{filename}.wav"
    os.rename(os.path.join(destination, default_filename),
              os.path.join(destination, new_filename))

    return os.path.join(destination, new_filename)

  except Exception as e:
    print("Error:", str(e))
    return None


def delete_file(file_path, type="Audio"):
  try:
    if os.path.exists(file_path):
      os.remove(file_path)
      print(f"{type} file deleted successfully.")
    else:
      print(f"{type} file does not exist.")

  except Exception as e:
    print(f"Error occurred while deleting the {type} file:", str(e))


def get_yt_video_ID(link):
  if 'youtube.com' in link:
    return link.split('v=')[1]
  elif 'youtu.be' in link:
    link = link.split('/')[-1]
    return link.split('?')[0]
  else:
    return None


# Add function that gets youtube_link and returns dictionary: 'title', 'description'
def get_data_from_youtube(video_url):
  yt = YouTube(video_url)

  title = yt.title
  author = yt.author
  channel_url = yt.channel_url
  publish_date = yt.publish_date
  # gen_data = generator(title, max_length=50, num_return_sequences=1)
  # content = gen_data[0]['generated_text']

  unique_filename = f'Transcript_{str(uuid.uuid4())[:6]}'

  try:
    data = YouTubeTranscriptApi.get_transcript(get_yt_video_ID(str(video_url)))
    content = ""
    for d in data:
      content += d['text'] + "; "
  except Exception as e:
    print("Error in the process of getting YT Video Transcript: ", e)
    content = "Something Went Wrong in the process of getting YT Video Transcript!"
  finally:
    with open(f'text/{unique_filename}.txt', 'w') as f:
      f.write(content)

  summary_text = get_summary(content)

  return {
    'title': title,
    'author': author,
    'channel_url': channel_url,
    'publish_date': publish_date,
    'content': unique_filename,
    'summary': summary_text
  }


def main():
  pass


if __name__ == '__main__':
  main()
