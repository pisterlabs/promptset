import json
import zipfile
import time

import pandas as pd
import openai

from config import OPENAI_KEY
from utils import get_files_with_extension

# Specify the zip file name
zip_file = "Sales Recordings-20230526T221129Z-001.zip"

# Create a ZipFile object
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
  # Extract all the contents of the zip file in the current directory
  zip_ref.extractall()

openai.api_key = OPENAI_KEY


def transcribe_file(file_path: str):
  """
    Function to transcribe the audio file.
    """
  print("Reading mp3 file...")
  with open(file_path, "rb") as audio_file:
    transcript = openai.Audio.transcribe("whisper-1", audio_file).text
  return transcript


mp3_file_list = get_files_with_extension('Sales Recordings', 'mp3')

transcripts = [transcribe_file(file) for file in mp3_file_list]

df = pd.DataFrame(transcripts, columns=['transcript'])
df.to_csv('transcript_sample.csv', index=False)

# df = pd.read_csv('transcript_sample.csv')

DEFAULT_MODEL = 'gpt-3.5-turbo'

SUMMARY_TEMPLATE_START = f"""your task is to answer 3 things from a customer support call for a company that sells poppy. \
1. Summary of the call transcript, in at most {num_words} words \
2. The name of the representative \
3. Did a purchase happen during the call? Only answer with Yes or No. \
Analyze the phone call transcript below, delimited by by triple backticks. \
```"""
SUMMARY_TEMPLATE_END = '''```\
Output in json format with Summary, Rep_name, Sale_status as keys.
```json '''

FEEDBACK_TEMPLATE_START = """your task is to analyze a customer service call for a company that sells poppy, \
to give feedback to the customer representative for 3 areas of improvements as a list \
focusing on any aspect that is about the sale. And output 3 quotes that are relevant as a list \
The phone call transcript is delimited by by triple dashes. \
Transcript:\
---"""
FEEDBACK_TEMPLATE_END = """---\
Output as a json object below, only includes Improvements and Improvement_Quotes as keys.\
```json """
QUESTION_TEMPLATE_START = """your task is to analyze a customer service call for a company that sells poppy, \
First, provide 3 areas that customer have questions/concerns about to the business owner, focusing on any aspects that \
is relevant to the sale, for example: Location of the puppy, Price of the puppy. etc. output as a list. \
Then find 3 example quotes that represents the questions/concerns, output as a list. \
The phone call transcript is delimited by by triple dashes. \
Transcript:\
---"""
QUESTION_TEMPLATE_END = """---\
Output as a json object below, only includes Concerns and Quotes as keys.\
```json
"""


def truncate_transcript(transcript):
  max_tokens = 4090
  num_words = 100
  # Create the start of your message
  message_start = SUMMARY_TEMPLATE_START.format(num_words=num_words)

  message_end = SUMMARY_TEMPLATE_END

  # Calculate the number of tokens left for the transcript
  tokens_for_start_end = (len(message_start) +
                          len(message_end)) // 4  # a rough estimation
  estimated_token_for_response = 300
  tokens_left = max_tokens - tokens_for_start_end - estimated_token_for_response

  # Now truncate the transcript
  estimated_characters_left = tokens_left * 4  # again, a rough estimation
  truncated_transcript = transcript[:estimated_characters_left]

  return message_start + truncated_transcript + message_end


def call_api(num_words: int,
             transcript: str,
             template_start: str,
             template_end: str,
             max_retries=5,
             delay=10) -> dict:
  TEMPLATE_START = template_start
  TEMPLATE_END = template_end
  for i in range(max_retries):
    try:
      response = openai.ChatCompletion.create(
        model=DEFAULT_MODEL,
        messages=[{
          "role": "user",
          "content": TEMPLATE_START + transcript + TEMPLATE_END
        }],
        stop=None,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
      )
      return response
    except (openai.error.APIError, openai.error.RateLimitError) as e:
      if i < max_retries - 1:
        print(
          f"APIError or RateLimitError encountered. Retrying in {delay} seconds..."
        )
        time.sleep(delay)
        delay *= 2
      else:
        raise e
    except openai.error.InvalidRequestError as e:
      if "maximum context length" in str(e):
        transcript = truncate_transcript(transcript)
      else:
        raise e
  return None


def generate_summary(num_words: int,
                     transcript: str,
                     max_retries=5,
                     delay=10) -> dict:
  response = call_api(num_words, transcript, SUMMARY_TEMPLATE_START,
                      SUMMARY_TEMPLATE_END, max_retries, delay)
  # Parse the response JSON
  json_result = response.choices[0].message.content
  try:
    result = json.loads(json_result)
  except json.JSONDecodeError:
    print("Failed to decode JSON for transcript:", json_result)
    result = {"Summary": None, "Rep_name": None, "Sale_status": None}

  return result


def generate_feedback(num_words: int,
                      transcript: str,
                      max_retries=5,
                      delay=10) -> dict:
  response = call_api(num_words, transcript, FEEDBACK_TEMPLATE_START,
                      FEEDBACK_TEMPLATE_END, max_retries, delay)
  # Parse the response JSON
  json_result = response.choices[0].message.content
  try:
    result = json.loads(json_result)
  except json.JSONDecodeError:
    print("Failed to decode JSON for transcript:", json_result)
    result = {"Improvements": None, "Improvement_Quotes": None}

  return result


def collect_customer_questions(num_words: int,
                               transcript: str,
                               max_retries=5,
                               delay=10) -> dict:
  response = call_api(num_words, transcript, QUESTION_TEMPLATE_START,
                      QUESTION_TEMPLATE_END, max_retries, delay)
  # Parse the response JSON
  json_result = response.choices[0].message.content
  try:
    result = json.loads(json_result)
  except json.JSONDecodeError:
    print("Failed to decode JSON for transcript:", json_result)
    result = {"Concerns": None, "Quotes": None}

  return result


if __name__ == "__main__":
  print("getting summary...")
  df['result'] = df['transcript'].apply(lambda x: generate_summary(100, x))
  df_summary = pd.json_normalize(df['result'].tolist())
  df = pd.concat([df, df_summary], axis=1)
  df.to_csv('transcript_sample.csv', index=False)

  print("getting questions...")
  df['question'] = df['transcript'].apply(
    lambda x: collect_customer_questions(100, x))
  df_question = pd.json_normalize(df['question'].tolist())
  df = pd.concat([df, df_question], axis=1)
  df.to_csv('transcript_sample.csv', index=False)

  print("getting feedback...")
  df['feedback'] = df['transcript'].apply(lambda x: generate_feedback(100, x))
  df_feedback = pd.json_normalize(df['feedback'].tolist())
  df = pd.concat([df, df_feedback], axis=1)
  df.to_csv('transcript_sample.csv', index=False)
