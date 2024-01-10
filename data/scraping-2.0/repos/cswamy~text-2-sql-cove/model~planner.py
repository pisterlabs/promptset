import openai
import os
from scripts import utils
from dotenv import load_dotenv

def build_plan(question:str, sql:str):
  """
  Build a plan for a given question and SQL query.
  """
  # Setup OpenAI API
  load_dotenv()
  openai.api_key = os.getenv("OPENAI_API_KEY")
  model = utils.get_model()

  # Build prompt
  messages = []

  # Build system instruction
  system_instruction = {
    "role": "system",
    "content": "You will be given a \"question\" in english and corresponding sql \"query\". Respond with a set of questions to check validity of \"query\". Generate as many \"questions\" as needed."
  }
  messages.append(system_instruction)

  # Build one-shot example
  one_shot_user = {
    "role": "user",
    "content": "\"question\": what were sales of cookware in california in dec 2019?\n\"query\": SELECT SUM(unitprice * orderquantity) AS cookware_revenues FROM salesorders so JOIN products pd ON so.productid = pd.productid JOIN storelocations sl ON so.storeid = sl.storeid WHERE pd.productname LIKE '%Cookware%' AND sl.statecode = 'CA' AND strftime('%Y-%m-%d', so.orderdate) BETWEEN '2019-12-01' AND '2019-12-31'"
  }
  messages.append(one_shot_user)
  one_shot_assistant = {
    "role": "assistant",
    "content": "does salesorders have columns unitprice and orderquantity? does products have column productname? does storelocations have column statecode? can productid be used to join salesorders and products? can storeid be used to join salesorders and storelocations? what column should be used to filter for cookware data? what column should be used to filter for california data? what column should be used to filter for dec 2019 data?"
  }
  messages.append(one_shot_assistant)

  # Build user question
  user_question = {
    "role": "user",
    "content": f"\"question\": {question}\n\"query\": {sql}"
  } 
  messages.append(user_question)

  # Get response from OpenAI API
  try:
    response = utils.call_openai(
    model=model['model_name'],
    messages=messages,
    temperature=model['temperature'],
    max_tokens=model['max_tokens'],
    top_p=model['top_p'],
    frequency_penalty=model['frequency_penalty'],
    presence_penalty=model['presence_penalty']
    )
  except:
    print('[ERROR] OpenAI API call failed for build plan.')
    return None

  # Return response
  return {
    'plan': response['choices'][0]['message']['content'],
    'input_tokens': response['usage']['prompt_tokens'],
    'output_tokens': response['usage']['completion_tokens']
    }

def execute_plan(target_schema:str, plan:str):
  """
  Execute plan for a given target schema and plan.
  """
  # Setup OpenAI API
  load_dotenv()
  openai.api_key = os.getenv("OPENAI_API_KEY")
  model = utils.get_model()

  # Build prompt
  messages = []

  # Build system instruction
  system_instruction = {
    "role": "system",
    "content": "For the database schema, answer the questions so responses can be used to build sql queries. Be concise." + target_schema
  }
  messages.append(system_instruction)

  # Build user question
  user_question = {
    "role": "user",
    "content": plan
  }
  messages.append(user_question)

  # Get response from OpenAI API
  try:
    response = utils.call_openai(
      model=model['model_name'],
      messages=messages,
      temperature=model['temperature'],
      max_tokens=model['max_tokens'],
      top_p=model['top_p'],
      frequency_penalty=model['frequency_penalty'],
      presence_penalty=model['presence_penalty']
      )
  except:
    print('[ERROR] OpenAI API call failed for execute plan.')
    return None

  # Parse response
  answers = response['choices'][0]['message']['content'].replace("\n", " ")
  qa = plan + " " + answers

  # Return response
  return {
    'qa': qa,
    'input_tokens': response['usage']['prompt_tokens'],
    'output_tokens': response['usage']['completion_tokens']
    }