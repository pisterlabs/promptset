#!/usr/bin/python3
from datetime import datetime
import pprint
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import networkx as nx
from networkx.algorithms import community
#from langchain.llms import OpenAI  # new way of import.
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI  # new way of import.

# Load the API key from an environment variable
api_key = os.environ.get("OPENAI_API_KEY")

# Check if the API key is available
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

pp=pprint.PrettyPrinter(indent=3,width=120)
print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")


# Load the txt file
txt_path = 'stateoftheunion.txt'
with open(txt_path, 'r') as f:
  txt = f.read()
print("0.-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("raw txt:")
print(txt)
print("Pres any key to continue..")
input()

# Get segments from txt by splitting on .
segments =  txt.split('.')
print("1.-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("segments =  txt.split('.')")
print(segments)
print("Pres any key to continue..")
input()

# Put the . back in
segments = [segment + '.' for segment in segments]
print("2.-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("segments = [segment + '.' for segment in segments]")
print(segments)
print("Pres any key to continue..")
input()

# Further split by comma
segments = [segment.split(',') for segment in segments]
print("3 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("segments = [segment.split(',') for segment in segments]")
print(segments)
print("Pres any key to continue..")
input()

# Flatten
segments = [item for sublist in segments for item in sublist]
print("4 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("segments = [item for sublist in segments for item in sublist]")
print(segments)
print("Pres any key to continue..")
input()


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
# It aims to create chunks of sentences based on the given chunk length and stride. The function uses the Pandas library for data manipulation.
# OVERLAP refers to the number of sentences to skip when creating the next chunk of sentences. It's a way to control the overlap between consecutive chunks.
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

sentences = create_sentences(segments, MIN_WORDS=20, MAX_WORDS=80)
print("5 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("sentences = create_sentences(segments, MIN_WORDS=20, MAX_WORDS=80)")
print(sentences)
print("Press any key to continue..")
input()

chunks = create_chunks(sentences, CHUNK_LENGTH=5, OVERLAP=1)
print("6 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("chunks = create_chunks(sentences, CHUNK_LENGTH=5, OVERLAP=1)")
print(chunks)
print("Press any key to continue..")
input()

chunks_text = [chunk['text'] for chunk in chunks]
print("7 -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
print("chunks_text = [chunk['text'] for chunk in chunks]")
print(chunks_text)
print("Press any key to continue..")
input()

def summarize_stage_1(chunks_text):

  print(f'Start time: {datetime.now()}')

  # Prompt to get title and summary for each chunk
  map_prompt_template = """Firstly, give the following text an informative title. Then, on a new line, write a 75-100 word summary of the following text:
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

# Run Stage 1 Summarizing
stage_1_outputs = summarize_stage_1(chunks_text)['stage_1_outputs']
# Split the titles and summaries
stage_1_summaries = [e['summary'] for e in stage_1_outputs]
stage_1_titles = [e['title'] for e in stage_1_outputs]
num_1_chunks = len(stage_1_summaries)

print("\n\n\n")
print (f"stage_1_outputs (all the titles)")
pp.pprint(stage_1_outputs)
print("Press any key to continue..")

print("\n\n\n")
print (f"stage_1_summaries (from the llm which summarized the chunk):\n\n\n")
pp.pprint(stage_1_summaries)
print("Press any key to continue..")
input()

print("\n\n\n")
print (f"stage_1_titles:\n\n\n")
pp.pprint(stage_1_titles)
print("Press any key to continue..")
input()

print("\n\n\n")
print (f"num_1_chunks:\n\n\n")
pp.pprint(num_1_chunks)
print("Press any key to continue..")
input()

# Use OpenAI to embed the summaries and titles. Size of _embeds: (num_chunks x 1536)
openai_embed = OpenAIEmbeddings()

# Use OpenAI's model to generate embeddings for each summary in stage_1_summaries
# Then convert the list of embeddings into a NumPy array for easier manipulation
summary_embeds = np.array(openai_embed.embed_documents(stage_1_summaries))
title_embeds = np.array(openai_embed.embed_documents(stage_1_titles))

# Get similarity matrix between the embeddings of the chunk summaries
summary_similarity_matrix = np.zeros((num_1_chunks, num_1_chunks))
summary_similarity_matrix[:] = np.nan

for row in range(num_1_chunks):
  for col in range(row, num_1_chunks):
    # Calculate cosine similarity between the two vectors
    similarity = 1- cosine(summary_embeds[row], summary_embeds[col])
    summary_similarity_matrix[row, col] = similarity
    summary_similarity_matrix[col, row] = similarity


# Draw a heatmap with the summary_similarity_matrix
plt.figure()
# Color scheme blues
plt.imshow(summary_similarity_matrix, cmap = 'Greens')
plt.savefig('heatmap.png')
plt.close('all')
print("saved: heatmap.png")
print("Press any key to continue..")
input()

# Run the community detection algorithm

def get_topics(title_similarity, num_topics = 8, bonus_constant = 0.25, min_size = 3):

  proximity_bonus_arr = np.zeros_like(title_similarity)
  for row in range(proximity_bonus_arr.shape[0]):
    for col in range(proximity_bonus_arr.shape[1]):
      if row == col:
        proximity_bonus_arr[row, col] = 0
      else:
        proximity_bonus_arr[row, col] = 1/(abs(row-col)) * bonus_constant

  title_similarity += proximity_bonus_arr

  title_nx_graph = nx.from_numpy_array(title_similarity)

  desired_num_topics = num_topics
  # Store the accepted partitionings
  topics_title_accepted = []

  resolution = 0.85
  resolution_step = 0.01
  iterations = 40

  # Find the resolution that gives the desired number of topics
  topics_title = []
  while len(topics_title) not in [desired_num_topics, desired_num_topics + 1, desired_num_topics + 2]:
    topics_title = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution)
    resolution += resolution_step
  topic_sizes = [len(c) for c in topics_title]
  sizes_sd = np.std(topic_sizes)
  modularity = community.modularity(title_nx_graph, topics_title, weight = 'weight', resolution = resolution)

  lowest_sd_iteration = 0
  # Set lowest sd to inf
  lowest_sd = float('inf')

  for i in range(iterations):
    topics_title = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution)
    modularity = community.modularity(title_nx_graph, topics_title, weight = 'weight', resolution = resolution)

    # Check SD
    topic_sizes = [len(c) for c in topics_title]
    sizes_sd = np.std(topic_sizes)

    topics_title_accepted.append(topics_title)

    if sizes_sd < lowest_sd and min(topic_sizes) >= min_size:
      lowest_sd_iteration = i
      lowest_sd = sizes_sd

  # Set the chosen partitioning to be the one with highest modularity
  topics_title = topics_title_accepted[lowest_sd_iteration]
  print(f'Best SD: {lowest_sd}, Best iteration: {lowest_sd_iteration}')

  topic_id_means = [sum(e)/len(e) for e in topics_title]
  # Arrange title_topics in order of topic_id_means
  topics_title = [list(c) for _, c in sorted(zip(topic_id_means, topics_title), key = lambda pair: pair[0])]
  # Create an array denoting which topic each chunk belongs to
  chunk_topics = [None] * title_similarity.shape[0]
  for i, c in enumerate(topics_title):
    for j in c:
      chunk_topics[j] = i

  return {
    'chunk_topics': chunk_topics,
    'topics': topics_title
    }

# Set num_topics to be 1/4 of the number of chunks, or 8, which ever is smaller
num_topics = min(int(num_1_chunks / 4), 8)
topics_out = get_topics(summary_similarity_matrix, num_topics = num_topics, bonus_constant = 0.2)
chunk_topics = topics_out['chunk_topics']
topics = topics_out['topics']

# Plot a heatmap of this array
plt.figure(figsize = (10, 4))
plt.imshow(np.array(chunk_topics).reshape(1, -1), cmap = 'tab20')
# Draw vertical black lines for every 1 of the x-axis 
for i in range(1, len(chunk_topics)):
  plt.axvline(x = i - 0.5, color = 'black', linewidth = 1.5)

# Save the figure as a PNG file
plt.savefig('heatmap_chunk_topics.png')
plt.close()
print("saved: heatmap_chunk_topics.png")
print("Press any key to continue..")
input()


def summarize_stage_2(stage_1_outputs, topics, summary_num_words = 250):
  print(f'Stage 2 start time {datetime.now()}')

  # Prompt that passes in all the titles of a topic, and asks for an overall title of the topic
  title_prompt_template = """Write an informative title that summarizes each of the following groups of titles. Make sure that the titles capture as much information as possible,
  and are different from each other:
  {text}

  Return your answer in a numbered list, with new line separating each title:
  1. Title 1
  2. Title 2
  3. Title 3

  TITLES:
  """

  map_prompt_template = """Wite a 75-100 word summary of the following text:
    {text}

    CONCISE SUMMARY:"""

  combine_prompt_template = 'Write a ' + str(summary_num_words) + """-word summary of the following, removing irrelevant information. Finish your answer:
  {text}
  """ + str(summary_num_words) + """-WORD SUMMARY:"""

  title_prompt = PromptTemplate(template=title_prompt_template, input_variables=["text"])
  map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
  combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

  topics_data = []
  for c in topics:
    topic_data = {
      'summaries': [stage_1_outputs[chunk_id]['summary'] for chunk_id in c],
      'titles': [stage_1_outputs[chunk_id]['title'] for chunk_id in c]
    }
    topic_data['summaries_concat'] = ' '.join(topic_data['summaries'])
    topic_data['titles_concat'] = ', '.join(topic_data['titles'])
    topics_data.append(topic_data)

  # Get a list of each community's summaries (concatenated)
  topics_summary_concat = [c['summaries_concat'] for c in topics_data]
  topics_titles_concat = [c['titles_concat'] for c in topics_data]

  # Concat into one long string to do the topic title creation
  topics_titles_concat_all = ''''''
  for i, c in enumerate(topics_titles_concat):
    topics_titles_concat_all += f'''{i+1}. {c}
    '''

  # print('topics_titles_concat_all', topics_titles_concat_all)

  title_llm = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo')
  title_llm_chain = LLMChain(llm = title_llm, prompt = title_prompt)
  title_llm_chain_input = [{'text': topics_titles_concat_all}]
  title_llm_chain_results = title_llm_chain.apply(title_llm_chain_input)


  # Split by new line
  titles = title_llm_chain_results[0]['text'].split('\n')
  # Remove any empty titles
  titles = [t for t in titles if t != '']
  # Remove spaces at start or end of each title
  titles = [t.strip() for t in titles]

  map_llm = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo')
  reduce_llm = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo', max_tokens = 4000)

  # Run the map-reduce chain
  docs = [Document(page_content=t) for t in topics_summary_concat]
  chain = load_summarize_chain(chain_type="map_reduce", map_prompt = map_prompt, combine_prompt = combine_prompt, return_intermediate_steps = True,
                              llm = map_llm, reduce_llm = reduce_llm)

  output = chain({"input_documents": docs}, return_only_outputs = True)
  summaries = output['intermediate_steps']
  stage_2_outputs = [{'title': t, 'summary': s} for t, s in zip(titles, summaries)]
  final_summary = output['output_text']

  # Return: stage_1_outputs (title and summary), stage_2_outputs (title and summary), final_summary, chunk_allocations
  out = {
    'stage_2_outputs': stage_2_outputs,
    'final_summary': final_summary
  }
  print(f'Stage 2 done time {datetime.now()}')

  return out


# Query GPT-3 to get a summarized title for each topic_data
out = summarize_stage_2(stage_1_outputs, topics, summary_num_words = 250)
stage_2_outputs = out['stage_2_outputs']
stage_2_titles = [e['title'] for e in stage_2_outputs]
stage_2_summaries = [e['summary'] for e in stage_2_outputs]
final_summary = out['final_summary']

print("\n\n\n")
print (f"stage_2_outputs:\n\n\n")
pp.pprint(stage_2_outputs)
print("Press any key to continue..")
input()

print("\n\n\n")
print (f"stage_2_titles:\n\n\n")
pp.pprint(stage_2_titles)
print("Press any key to continue..")
input()

print("\n\n\n")
print (f"stage_2_summaries:\n\n\n")
pp.pprint(stage_2_summaries)
print("Press any key to continue..")
input()

print("\n\n\n")
print (f"final_summary:\n\n\n")
pp.pprint(final_summary)


