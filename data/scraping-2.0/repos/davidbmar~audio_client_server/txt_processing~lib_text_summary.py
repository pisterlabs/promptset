#!/usr/bin/python3

from datetime import datetime
import pprint
import pandas as pd
import numpy as np
import json
import os
import re
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain


# Import other necessary libraries and modules

def create_sentences(segments, MIN_WORDS, MAX_WORDS):

  # Combine the non-sentences together
  sentences = []

  is_new_sentence = True
  sentence_length = 0
  sentence_num = 0
  sentence_segments = []

  for i in range(len(segments)):
    if is_new_sentence == True:
      is_new_sentence = False
    # Append the segment
    sentence_segments.append(segments[i])
    segment_words = segments[i].split(' ')
    sentence_length += len(segment_words)

    # Check Conditions: If the sentence length is at least MIN_WORDS and the segment ends with a period, 
    # or if the sentence length exceeds MAX_WORDS, the sentence is considered complete.
    # If exceed MAX_WORDS, then stop at the end of the segment. Only consider it a sentence if the length is at least MIN_WORDS.
    if (sentence_length >= MIN_WORDS and segments[i][-1] == '.') or sentence_length >= MAX_WORDS:
      sentence = ' '.join(sentence_segments)
      sentences.append({
        'sentence_num': sentence_num,
        'text': sentence,
        'sentence_length': sentence_length
      })
      # Reset
      is_new_sentence = True
      sentence_length = 0
      sentence_segments = []
      sentence_num += 1

  return sentences

# The function create_chunks takes a list of sentences, a CHUNK_LENGTH, and a OVERLAP as arguments. 
# It aims to create chunks of sentences based on the given chunk length and stride. The function uses the Pandas library for data manipul
# OVERLAP refers to the number of sentences to skip when creating the next chunk of sentences. It's a way to control the overlap between 
# How Stride Works: 
#    Let's say you have a CHUNK_LENGTH of 5 and a OVERLAP of 2. If your first chunk starts at sentence 0 and ends at sentence 4, then:
#    With a OVERLAP of 2, the next chunk will start at sentence 3 (i.e., 5 - 2 = 3) and end at sentence 7.
#    The third chunk will then start at sentence 6 (i.e., 8 - 2 = 6) and so on.
def create_chunks(sentences, CHUNK_LENGTH, OVERLAP):
  sentences_df = pd.DataFrame(sentences)

  chunks = []
  for i in range(0, len(sentences_df), (CHUNK_LENGTH - OVERLAP)):
    chunk = sentences_df.iloc[i:i+CHUNK_LENGTH]
    chunk_text = ' '.join(chunk['text'].tolist())

    chunks.append({
      'start_sentence_num': chunk['sentence_num'].iloc[0],
      'end_sentence_num': chunk['sentence_num'].iloc[-1],
      'text': chunk_text,
      'num_words': len(chunk_text.split(' '))
    })

  chunks_df = pd.DataFrame(chunks)
  return chunks_df.to_dict('records')

def parse_title_summary_results(results):
  out = []
  for e in results:
    e = e.replace('\n', '')
    if '|' in e:
      processed = {'title': e.split('|')[0],
                    'summary': e.split('|')[1][1:]
                    }
    elif ':' in e:
      processed = {'title': e.split(':')[0],
                    'summary': e.split(':')[1][1:]
                    }
    elif '-' in e:
      processed = {'title': e.split('-')[0],
                    'summary': e.split('-')[1][1:]
                    }
    else:
      processed = {'title': '',
                    'summary': e
                    }
    out.append(processed)
  return out

def summarize_stage_1(chunks_text):

  print(f'Start time: {datetime.now()}')

  # Prompt to get title and summary for each chunk
  map_prompt_template = """Firstly, give the following text an informative title. Then, on a new line, write a 75-100 word summary of the
  {text}

  Return your answer in the following format:
  Title | Summary...
  e.g.
  Why Artificial Intelligence is Good | AI can make humans more productive by automating many repetitive processes.

  TITLE AND CONCISE SUMMARY:"""

  map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

  # Define the LLMs
  map_llm = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo')
  map_llm_chain = LLMChain(llm = map_llm, prompt = map_prompt)

  # Initialize an empty list to store the outputs
  stage_1_outputs = []

  # Loop through each chunk and process it
  for chunk in chunks_text:
    single_prompt_input = {'text': chunk}
    single_prompt_result = map_llm_chain.apply([single_prompt_input])
    parsed_result = parse_title_summary_results([e['text'] for e in single_prompt_result])
    stage_1_outputs.extend(parsed_result)

  print(f'Stage 1 done time {datetime.now()}')
  return {
    'stage_1_outputs': stage_1_outputs
  }


def summarize_text(input_text,name_of_object):

   # Load the API key from an environment variable
   api_key = os.environ.get("OPENAI_API_KEY")
   # Check if the API key is available
   if api_key is None:
       raise ValueError("OPENAI_API_KEY environment variable not set.")

   pp=pprint.PrettyPrinter(indent=3,width=120)

   # Join the list into a single string
   segments = [sentence.strip() for sentence in input_text.split('.') if sentence.strip()]

   # Put the . back in
   segments = [segment + '.' for segment in segments]

   # Further split by ? 
   #segments = [segment.split('?') for segment in segments]
   segments = [sub_segment for segment in segments for sub_segment in re.split(r'(?<=\?)', segment) if sub_segment.strip()]

   # Further split by comma
   segments = [segment.split(',') for segment in segments]

   segments = [item for sublist in segments for item in sublist]

   sentences = create_sentences(segments, MIN_WORDS=20, MAX_WORDS=80)
   chunks = create_chunks(sentences, CHUNK_LENGTH=5, OVERLAP=1)
   chunks_text = [chunk['text'] for chunk in chunks]

   # Run Stage 1 Summarizing
   stage_1_outputs = summarize_stage_1(chunks_text)['stage_1_outputs']
   # Split the titles and summaries
   stage_1_summaries = [e['summary'] for e in stage_1_outputs]
   stage_1_titles = [e['title'] for e in stage_1_outputs]
   num_1_chunks = len(stage_1_summaries)

   # Combine titles and summaries into a list of dictionaries
   combined_output = [{"title": title, "summary": summary} for title, summary in zip(stage_1_titles, stage_1_summaries)]

   # Convert the list of dictionaries to a JSON string
   json_output = json.dumps(combined_output, indent=3)

   with open(f'./temp/{name_of_object}', 'w') as file:
       file.write(json_output)
       print (f"Wrote ./temp/{name_of_object}")

   # Return the JSON string
   return json_output

def generate_summary_html(json_data,name_of_object):
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Summary Display</title>
        <style>
            .summary {
                margin-left: 20px;
                display: block; /* Default to showing */
            }
            .title {
                font-weight: bold;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <div id="content">
    '''

    for item in json_data:
        html_content += f'''
            <div class="title" onclick="toggleSummary(this)">{item['title']}</div>
            <div class="summary">{item['summary']}</div>
        '''

    html_content += '''
        </div>
        <script>
            function toggleSummary(element) {
                var summary = element.nextElementSibling;
                summary.style.display = summary.style.display === 'none' ? 'block' : 'none';
            }
        </script>
    </body>
    </html>
    '''

    # Save the HTML content to a file
    with open(f'./temp/{name_of_object}.html', 'w') as file:
       file.write(html_content)
       print (f"Wrote ./temp/{name_of_object}.html")

    return html_content

def setup():
    directory = "temp"  # Replace with your desired path

    if not os.path.exists(directory):
       os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    input_text="""
      QSBS is a tax treatment for qualified small business stock.
      What that means is that if you're a business that fits like these five criteria, which
      most tech companies fit, then when you sell, if you've held the stock for five years, or
      even if it's a little bit less than that, you can still kind of roll it over, your first
      $10 million of your gain are going to be tax free.
      Not only most tech companies, most new C-corps that hold for five years would fall under
      this.
    """

    setup()
    json_output = summarize_text(input_text)
    json_data = json.loads(json_output)
    html_content = generate_summary_html(json_data)


