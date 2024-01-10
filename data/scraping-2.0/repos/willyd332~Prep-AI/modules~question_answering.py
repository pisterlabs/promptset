import os
import openai
from dotenv import load_dotenv
from modules.training_data.question_answer_pairs import cold_war_QA_pairs, cold_war_context

# Load env variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# Answers Class
class Answers():
  '''
  Create a class that can be used to answer questions about an uploaded text
  '''
  def __init__(self, path_to_jsonl):
    '''
    Takes in path to jsonl file
    Uploads it to OpenAI and prepares it to answer questions
    '''
    print("Answers Initialized!")
    jsonl_file = open(path_to_jsonl)
    self.filename = os.path.basename(path_to_jsonl).split(".")[0]
    self.file_id = self.upload_jsonl(jsonl_file)
    print(self.file_id)



  def upload_jsonl(self, jsonl_file):
    '''
    Takes in the jsonl file
    Uploads it to Open AI
    Returns the file id
    '''
    response = openai.File.create(file=jsonl_file, purpose='answers')
    return response["id"]

  @staticmethod
  def submit_query(question, fileid, temperature = 0.1):
    '''
    Takes in question string and suggested temperature
    Queries Open AI for the answer
    Returns answer as string
    '''
    response = openai.Answer.create(
      search_model="ada",
      model="ada",
      temperature = temperature, # How risky the model will be in generating answers
      question=question,
      file=fileid,
      examples_context=cold_war_context,
      examples=cold_war_QA_pairs,
      max_tokens=200,
      stop=["\n", "<|endoftext|>"],
      n=3, # This means that it will generate 3 different answers!
      return_prompt=True # This will return the prompt used by Open AI
    )
    return response