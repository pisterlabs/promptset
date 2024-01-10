
from dotenv import load_dotenv, find_dotenv
import os
import wikipedia
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import SimpleSequentialChain


_ = load_dotenv(find_dotenv())  # read local .env file
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_page_content_from_wikipedia(page_name:str):
  """Get page content from a wikipedia page."""
  input = wikipedia.page(page_name, auto_suggest=False)
  return input.content


def create_summarisation_chain(openai_api_key:str, model_name:str="gpt-3.5-turbo", temperature:float=0, verbose:bool=False):
  """Summarisation chain: create a summarisation chain using load_summarize_chain function from langchain with map_reduce chain type. 
  This chain recieves chunks of text and create a summary for each step (map step) then concatenates all 
  summaries in a single, concise summary (reduce step)."""

  map_prompt = """
  You are a bird enthusiast who has a podcast about birds. Given a text about a bird, extract some key information about this bird and curiosities that you could share on your next podcast.
  Also include cultural curiosities about the bird if mentioned in the text.
  Text: "{text}"
  Highlights:
  """
  map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

  reduce_prompt = """
  You will be given with summary with facts about a bird. Divide these facts into main topics, and provide the output in the format of a list, something like:

  Topic 1:
  - Highlight 1
  - Highlight 2
  - Highlight 3

  Summary: {text}
  """
  reduce_prompt_template = PromptTemplate(template=reduce_prompt, input_variables=["text"])

  llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=openai_api_key)
  summary_chain = load_summarize_chain(llm=llm,
                                      chain_type='map_reduce',
                                      map_prompt=map_prompt_template,
                                      combine_prompt=reduce_prompt_template,
                                      verbose=verbose
                                      )
  return summary_chain



def create_dialogue_chain(podcast_name, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=1.3, verbose=False):
  """Dialogue chain: create a function chain using create_structured_output_chain function from langchain.
  This will recieve a text summary and will generate a podcast dialogue between two people from it. 
   The output will be a python dict with the dialogue. """
  prompt_dialogue = ChatPromptTemplate.from_messages(
      [
          ("system", f"""Generate the script of a podcast episode between Mark and Anna. Mark is a bird enthusiast and is the host of the podcast. Anna is a bird expert who came to the episode to discuss about a bird. Given a text with facts about this bird, create a conversation between them, discussing the facts present in the text. At the begining of the podcast, make them introduce themselves initially. Make the dialogue casual, funny and informative. Avoid repetetive expressions or repeting the name of the bird to many times. The name of the podcast is '{podcast_name}'"""),
          ("human", "{bird_facts}")
      ]
  )

  json_schema = {
      "title": "Dialogue",
      "description": "Creating a dialogue between two people.",
      "type": "object",
      "properties": {
          "podcast_dialogues": {
              "type": "array",
              "description": "An array of podcast dialogues containing the speaker name and their dialogue or text",
              "items": {
                  "type": "object",
                  "properties": {
                      "speaker_name": {
                          "type": "string",
                          "description": "The name of the person who is speaking in the podcast. Should be Mark, the host, or Anna, the specialist in birds."},
                      "speaker_text": {
                          "type": "string",
                          "description": "The speciic dialogue or text spoken by the person"}
                      }
                  }
              },
          },
      "required": ["podcast_dialogues"],
      }


  llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=openai_api_key)
  dialogue_chain = create_structured_output_chain(output_schema=json_schema, prompt=prompt_dialogue, llm=llm, verbose=verbose)
  return dialogue_chain


def split_text_into_documents(input_text:str):
  """Split the text into chunks of 2000 tokens."""
  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
      chunk_size = 2000,
      chunk_overlap  = 0,
      is_separator_regex = False,
      separators = ['\n\n\n='] # splitter in case of wikipedia, should breake on new sections.
  )

  texts = text_splitter.create_documents([input_text])
  print(f"The text was splited into {len(texts)} chunks of texts.")
  return texts

## SEQUENTIAL CHAIN - outputs dialogue
def create_podcast_dialogue_from_text(input_text:str, podcast_name:str, verbose:bool=False):
  """Sequential Chain: create a sequential chain with the summary chain and dialogue chain. 
  This will take a text of any size and it will generate a dictionary with a podcast episode 
  with 2 people discussing the topics of the text."""
  texts = split_text_into_documents(input_text)

  summary_chain = create_summarisation_chain(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0, verbose=verbose)
  dialogue_chain = create_dialogue_chain(podcast_name, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=1.3, verbose=verbose)

  overall_chain = SimpleSequentialChain(chains=[summary_chain, dialogue_chain], verbose=verbose)
  podcast_dialogue = overall_chain.run(texts)
  return podcast_dialogue