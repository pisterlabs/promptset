from create_chunks import create_sentences
from create_chunks import create_chunks
from create_chunks import parse_title_summary_results
from create_chunks import segments
from datetime import datetime
import matplotlib.pyplot as plt
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
import os

api_key = os.getenv("OPENAI_API_KEY")

sentences = create_sentences(segments, MIN_WORDS=20, MAX_WORDS=80)
chunks = create_chunks(sentences, CHUNK_LENGTH=5, STRIDE=1)
chunks_text = [chunk['text'] for chunk in chunks]

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
  map_llm = OpenAI(temperature=0, model_name = 'text-davinci-003')
  map_llm_chain = LLMChain(llm = map_llm, prompt = map_prompt)
  map_llm_chain_input = [{'text': t} for t in chunks_text]
  # Run the input through the LLM chain (works in parallel)
  map_llm_chain_results = map_llm_chain.apply(map_llm_chain_input)

  stage_1_outputs = parse_title_summary_results([e['text'] for e in map_llm_chain_results])

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

with open('stage_1_summaries.txt', 'w') as f:
  for item in stage_1_summaries:
    f.write("%s\n" % item)

with open('stage_1_titles.txt', 'w') as f:
  for item in stage_1_titles:
    f.write("%s\n" % item)

with open('stage_1_outputs.txt', 'w') as f:
  for item in stage_1_outputs:
    f.write("%s\n" % item)  

