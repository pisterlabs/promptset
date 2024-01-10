from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine
import networkx as nx
from networkx.algorithms import community
from recursive_summary_stage1 import stage_1_summaries, stage_1_titles, num_1_chunks

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
import os

api_key = os.getenv("OPENAI_API_KEY")

with open("stage_1_summaries.txt", "r") as file:
    stage_1_summaries = file.read()

with open("stage_1_titles.txt", "r") as file:
    stage_1_titles = file.read()

with open("stage_1_outputs.txt", "r") as file:
    stage_1_outputs = file.read()


# Use OpenAI to embed the summaries and titles. Size of _embeds: (num_chunks x 1536)
openai_embed = OpenAIEmbeddings()

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


def summarize_stage_2(stage_1_outputs, topics, summary_num_words=250):
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

    map_prompt_template = """Write a 75-100 word summary of the following text:
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
            'summaries': [stage_1_outputs[int(chunk_id)]['summary'] for chunk_id in c],
            'titles': [stage_1_outputs[int(chunk_id)]['title'] for chunk_id in c]
        }
        topic_data['summaries_concat'] = ' '.join(topic_data['summaries'])
        topic_data['titles_concat'] = ', '.join(topic_data['titles'])
        topics_data.append(topic_data)

    # Get a list of each community's summaries (concatenated)
    topics_summary_concat = [c['summaries_concat'] for c in topics_data]
    topics_titles_concat = [c['titles_concat'] for c in topics_data]

    # Concat into one long string to do the topic title creation
    topics_titles_concat_all = ''
    for i, c in enumerate(topics_titles_concat):
        topics_titles_concat_all += f'''{i+1}. {c}
    '''

    # print('topics_titles_concat_all', topics_titles_concat_all)

    title_llm = OpenAI(temperature=0, model_name='text-davinci-003')
    title_llm_chain = LLMChain(llm=title_llm, prompt=title_prompt)
    title_llm_chain_input = [{'text': topics_titles_concat_all}]
    title_llm_chain_results = title_llm_chain.apply(title_llm_chain_input)

    # Split by new line
    titles = title_llm_chain_results[0]['text'].split('\n')
    # Remove any empty titles
    titles = [t for t in titles if t != '']
    # Remove spaces at start or end of each title
    titles = [t.strip() for t in titles]

    map_llm = OpenAI(temperature=0, model_name='text-davinci-003')
    reduce_llm = OpenAI(temperature=0, model_name='text-davinci-003', max_tokens=-1)

    # Run the map-reduce chain
    docs = [Document(page_content=t) for t in topics_summary_concat]
    chain = load_summarize_chain(chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt,
                                 return_intermediate_steps=True, llm=map_llm, reduce_llm=reduce_llm)

    output = chain({"input_documents": docs}, return_only_outputs=True)
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

with open('stage_2_titles.txt', 'w', encoding='utf-8') as f:
    for title in stage_2_titles:
        f.write(title + '\n')

with open('stage_2_summaries.txt', 'w', encoding='utf-8') as f:
    for summary in stage_2_summaries:
        f.write(summary + '\n\n')  

with open('final_summary.txt', 'w', encoding='utf-8') as f:
    f.write(final_summary)