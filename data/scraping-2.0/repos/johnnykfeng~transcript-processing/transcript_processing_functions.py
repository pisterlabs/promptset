from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import MarkdownTextSplitter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import json
import os
import openai
import streamlit as st
import time

# disable api key for the streamlit app
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# openai.api_key = OPENAI_API_KEY

#instantiate chat model
# chat = ChatOpenAI(
#     # openai_api_key=OPENAI_API_KEY ,
#     temperature=0,
#     model='gpt-3.5-turbo')

# chat16k = ChatOpenAI(
#     # openai_api_key=OPENAI_API_KEY ,
#     temperature=0,
#     model='gpt-3.5-turbo-16k')

import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return result
    return wrapper

"""# Part **1**: Processing raw transcript"""

import tiktoken
price_gpt35_turbo = 0.002 # $0.002/1k tokens
def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


import docx

def extract_text_from_docx(file_path):
    """Extracts text from a docx file."""
    doc = docx.Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

def extract_text_from_plaintext(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data



def transcript_token_size(raw_transcript, verbose = True):
  char_len = len(raw_transcript)
  token_len = num_tokens_from_string(raw_transcript)
  if verbose:
    print(f'Character length in raw transcript: {char_len}')
    print(f'Number of tokens in raw transcript: {token_len}')
    print(f'Char_len/token_len: {char_len/token_len:.2f}')
    print(f'Cost of input prompt with gpt-3.5-turbo: ${token_len*0.002/1000}')

  return token_len

"""## Splitting raw transcript

ChatGPT models have a token limit. For GPT3.5-turbo, the limit is 4096 tokens [(docs)](https://platform.openai.com/docs/models/gpt-3-5). Most transcripts exceed that, so it must be split into chunks.
"""

def transcript_splitter(raw_transcript, chunk_size=10000, chunk_overlap=200):
  markdown_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  transcript_docs = markdown_splitter.create_documents([raw_transcript])
  return transcript_docs

def transcript2essay(transcript, chat_model):
  system_template = "You are a helpful assistant that summarizes a transcript of podcasts or lectures."
  system_prompt = SystemMessagePromptTemplate.from_template(system_template)
  # human_template = "Summarize the main points of this presentation's transcript: {transcript}"
  human_template = """Rewrite the contents and information of the presentation into a well written essay.\
  Write the essay as if the speaker wrote it himself from the same knowledge he used to create the presentation. \
  Include the speaker's full name in the essay and refer to him/her with the full name. \
  Also include the names of the people who asked questions and the questions they asked. \
  The transcript of this presentation is delimited in triple backticks:
  ```{transcript}```"""
  human_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

  result = chat_model(chat_prompt.format_prompt(transcript=transcript).to_messages())
  return result.content

def create_essay_parts(transcript_docs, chat_model):
  essay_response=''
  for i, text in enumerate(transcript_docs):
    essay = transcript2essay(text.page_content, chat_model)
    essay_response = f'\n\n#Part {i+1}\n'.join([essay_response, essay])

  return essay_response

def merge_essays(essays, chat_model):
  system_template = """You are a helpful assistant that summarizes and \
    processes large text information."""
  system_prompt = SystemMessagePromptTemplate.from_template(system_template)

  human_template = """Consolidate the multiple parts of the text into one \
  coherent essay or article that accurately captures the content of the multiple\
  parts without losing any information. Make sure to include the speaker's full name \
  and the questions asked by the audience as well as the response to those questions. \
  The entire text is delimited in triple backticks and the parts are divided by
  #heading:\n
  ```{essays}```"""
  human_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

  final_essay = chat_model(chat_prompt.format_prompt(essays=essays).to_messages())

  return final_essay.content

# @timer_decorator
def full_transcript2essay(raw_transcript:str, chat_model, verbose=True):
  print('Chunking transcript...')
  transcript_docs = transcript_splitter(raw_transcript)
  t1 = time.time()
  print('Creating essay parts...')
  essay_parts = create_essay_parts(transcript_docs, chat_model)
  t2 = time.time()-t1
  print('Merging essay parts...')
  t1 = time.time()
  final_essay = merge_essays(essay_parts, chat_model)
  t3 = time.time()-t1
  if verbose:
    print(f'Created essay parts in {t2:.2f} seconds')
    print(f'Merged essay parts in {t3:.2f} seconds')
  return final_essay


# """# Part 2: Extracting from essay"""

# Extracting from generated essay
def extract_topics_from_text(text, user_prompt, chat_model):
  system_template = """You are a helpful assistant that preprocesses text, \
                      writings and presentation transcripts"""
  system_prompt = SystemMessagePromptTemplate.from_template(system_template)

  human_template = """{user_prompt}
  The text is delimited in triple backticks:
  ```{text}```"""
  human_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

  result = chat_model(chat_prompt.format_prompt(text=text, user_prompt=user_prompt).to_messages())
  return result.content



def extract_metadata_as_json(essay, chat_model):

  system_template = """ Given the essay delimited in triple backticks, generate and extract important \
  information such as the title, speaker, summary, a list of key topics, \
  and a list of important takeaways for each topic. \
  Format the response as a JSON object, with the keys 'Title', 'Topics', 'Speaker', \
  'Summary', and 'Topics' as the keys and each topic will be keys for list of takeaways. \
  Example of JSON output: \n \
 {{\
  'Title': 'Title of the presentation',\
  'Speaker': 'John Smith',\
  'Summary': 'summary of the presentation',\
  'Topics': [\
  {{\
  'Topic': 'topic 1',\
  'Takeaways': [\
  'takeaway 1',\
  'takeaway 2',\
  'takeaway 3'\
  ]\
  }},\
  {{\
  'Topic': 'topic 2',\
  'Takeaways': [\
  'takeaway 1',\
  'takeaway 2',\
  'takeaway 3'\
  ]\
  }},\
  {{\
  'Topic': 'topic 3',\
  'Takeaways': [\
  'takeaway 1',\
  'takeaway 2',\
  'takeaway 3'\
  ]\
  }},\
  {{\
  'Topic': 'topic 4',\
  'Takeaways': [\
  'takeaway 1',\
  'takeaway 2',\
  'takeaway 3'\
  ]\
  }}\
  ]\
  }}"""
  
  system_prompt = SystemMessagePromptTemplate.from_template(system_template)

  human_template = """Essay: ```{text}```"""
  
  human_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

  result = chat_model(chat_prompt.format_prompt(text=essay).to_messages())
  try:
    metadata_json = json.loads(result.content)
  except Exception as e:
    print(e)
    metadata_json = result.content  
  return metadata_json


# """#Part 3: Experimental"""

def generate_qa(text, chat_model):
  system_template = """You are a helpful assistant that preprocesses text, \
                      writings and presentation transcripts in the context of \
                      large-language models and machine learning research"""
  system_prompt = SystemMessagePromptTemplate.from_template(system_template)

  human_template = """Given the article, rewrite it in a question and answer \
  format, where questions are asked about the important takeaways of the article \
  and detailed answers are provided based on the content of the article. The \
  goal is to present the same important information of the article. \
  Format the questions and answers as such:
  Q. question...
  A. answer...
  The article is delimited in triple backticks:
  ```{text}```"""
  human_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

  result = chat_model(chat_prompt.format_prompt(text=text).to_messages())
  return result.content

def generate_mc_questions(text, chat_model):
  system_template = """Given the essay delimited in triple backticks, \
  generate 5 multiple choice questions based on the contents of the essay. \
  The goal of the these questions is to  quiz the audience after who have \
  read or listen to the essay. Format the quiz as follows: 
  Q. question... \n \
  a. choice 1 \n \
  b. choice 2 \n \
  c. choice 3 \n \
  d. choice 4 \n \
  """
  system_prompt = SystemMessagePromptTemplate.from_template(system_template)

  human_template = """Essay: \n```{text}```"""
  human_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

  result = chat_model(chat_prompt.format_prompt(text=text).to_messages())
  return result.content

def generate_answers(text, chat_model):
  system_template = """Given the multiple choice questions and the \
  essay they were generated from, generate the answers to the questions. \
  Make sure the answers are in the same order as the questions and provide \
  an explanation for each answer based on the contents of the essay. \
  Format the answers as follows: \n \
  
  """
  system_prompt = SystemMessagePromptTemplate.from_template(system_template)

  human_template = """Essay: \n```{text}```"""
  human_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

  result = chat_model(chat_prompt.format_prompt(text=text).to_messages())
  return result.content


def json2rst(metadata, rst_filepath):
  if not isinstance(metadata, dict):
      metadata = json.loads(metadata)
  
  # rst_filepath = './essays/test.rst'
  with open(rst_filepath, 'w') as the_file:
      the_file.write("\n")
      for key, value in metadata.items():
          if key == "Title":
              title_mark = "=" * len(f'{value}')
              the_file.write(title_mark + '\n')
              the_file.write(f"{value} \n")
              the_file.write(title_mark + '\n')
          elif key == "Speaker":
              the_file.write('*' + f"{value}" + '* \n\n')
          elif key == "Summary":
              title_mark = '-' * len(f'{key}')
              the_file.write("Summary \n")
              the_file.write(title_mark + '\n')
              the_file.write(f"{value} \n\n")
          elif key == "Topics":
              the_file.write("Topics: \n")
              the_file.write(title_mark + '\n')
              for topic in value:
                  the_file.write("\t" + f"{topic['Topic']} \n")
                  for takeaway in topic['Takeaways']:
                      the_file.write("\t\t" + f"* {takeaway} \n")
      the_file.write("\n")


def mc_question_json(text, chat_model, n=5):
  system_template = """Given the corpus of text, \
  generate {n} multiple choice questions\
  based on the contents of the text. The goal of the these questions is to \
  quiz the audience after who have read the text. Make sure to randomize \
  the order of the answers for each question and evenly distribute the correct \
  answer across the options. Each question should be different and not repeated. \
  Format the questions in JSON as follows, make sure to use double quotes:\n \
  {{\
    "questions": [\
      {{\
        "question": "Who did X?",\
        "options": [\
         "A) Answer 1",\
         "B) Answer 2",\
         "C) Answer 3",\
         "D) Answer 4"
        ],\
        "correct_answer": "C) Answer 3", \
        "explanation": "Explanation of the correct answer" \
      }},\
      // More questions...\
    ]\
  }}
  """
  system_prompt = SystemMessagePromptTemplate.from_template(system_template)
  human_template = """The text delimited in triple backticks:
  ```{text}```"""
  human_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
  result = chat_model(chat_prompt.format_prompt(n=n, text=text).to_messages())
  try:
    json_result = json.loads(result.content)
  except Exception as e:
    print(e)
    json_result = result.content
  return json_result


def mc_question_json_v2(text, chat_model, n=5):
  system_template = """Given the corpus of text, \
  generate {n} multiple choice questions\
  based on the contents of the text. The goal of the these questions is to \
  quiz the audience after who have read the text. Make sure to randomize \
  the order of the answers for each question. \
  Format the questions in JSON as follows, make sure to use double quotes: \
  {{\
    "questions": [\
      {{\
        "question": "Who did X?",\
        "options": [\
          {{"A": "Answer 1"}},\
          {{"B": "Answer 2"}},\
          {{"C": "Answer 3"}},\
          {{"D": "Answer 4"}\
        ],\
        "correct_answer": {{"C": "Answer 3"}}, \
        "explanation": "Explanation of the correct answer" \
      }},\
      // More questions...\
    ]\
  }}
  """
  system_prompt = SystemMessagePromptTemplate.from_template(system_template)
  human_template = """The text delimited in triple backticks:
  ```{text}```"""
  human_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
  result = chat_model(chat_prompt.format_prompt(n=n, text=text).to_messages())
  try:
    json_result = json.loads(result.content)
  except Exception as e:
    print(e)
    json_result = result.content
  return json_result
