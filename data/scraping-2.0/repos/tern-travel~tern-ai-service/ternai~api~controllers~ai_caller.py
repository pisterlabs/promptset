from openai import OpenAI
import collections.abc
import json
from api.models import ExternalCall, APICall, Prompt, AIModel


def make_ai_call(prompt:str, endpoint_used:ExternalCall, force_max_context:bool):
  
  client = OpenAI()
  model_to_use = ""

  if force_max_context: #used for long prompts.
    largest_model = AIModel.objects.get(largest_token_model=True)    
    model_to_use = largest_model.api_value

  else:
    model_to_use = endpoint_used.endpoint.prompt.model.api_value, #We get the endpoint to use from the API
    model_to_use = model_to_use[0]

  messages=[
          {"role": "system", "content": endpoint_used.endpoint.prompt.instructions},
          {"role": "user", "content": prompt}
        ]

  response = client.chat.completions.create(
    model= model_to_use,
    temperature=endpoint_used.endpoint.prompt.ai_temperature,
    response_format={ "type": "json_object" }, #NOTE THE JSON OBJECT TYPE
    messages = messages
  )

  ai_response_dict, valid_response, reason = response_validation(endpoint_used.endpoint.prompt, response.choices[0].message.content)
  
  if valid_response:
    #The response we got back from OpenAI is valid
    response_dict = {}
    response_dict['status'] = "Success"
    response_dict['response'] = ai_response_dict
    log_api_call(endpoint_used.endpoint.prompt.instructions,response_dict, endpoint_used, True, "None")
    return response_dict

  else:
    #invalid response
    error_response = {}
    error_response['status'] = "Error"
    error_response['reason'] = reason
    response_dict['response'] = ai_response_dict
    return response_dict

#This checks to make sure the JSON we get back is valid and the keys we expect are passed back 
def response_validation(source_prompt:Prompt, response_text:str):

  #First we'll try to load the JSON into a dictionary. 
  try: 
    response = json.loads(response_text)
  except:
    #there was an error loading the JSON
    return response, False, "Invalid JSON Returned from OpenAI. Could not load into Dictionary."   


  #Now we'll validate that the keys required are present. 
  required_keys_dict = source_prompt.required_response_keys

  if required_keys_dict == None:
    return response, True, ""

  req_keys = list(required_keys_dict.keys())

  #For now we only validate the keys are somewhere in the response, not the tree structure 


  given_keys = key_collector(response)

  if set(req_keys).issubset(set(given_keys)) == False:
    #We're missing keys
    return response, False, "Missing Keys. Required keys were not provided from OpenAI."   
  

  return response, True, ""

# A little recursive function to get all the keys from the dictionary
def key_collector(dict_to_extract:dict): 
  given_keys = []
  for key in dict_to_extract:
    given_keys.append(key)
    
    if isinstance(dict_to_extract[key], dict):
      given_keys += key_collector(dict_to_extract[key])

    #If there is an array of dictionaries we want to check that too
    if isinstance(dict_to_extract[key],collections.abc.Sequence):
      for item in dict_to_extract[key]:
        if isinstance(item, dict):
          given_keys += key_collector(item)

  
  return given_keys

#This handles creating a new log record for AI calls. 
def log_api_call(prompt:str, results:dict, caller:ExternalCall, valid_call:bool, error:str=""):
  
  response_json = json.dumps(results)

  new_api_call = APICall(
    prompt_text=prompt,
    in_progress=False,
    complete=True,
    response=response_json, 
    source_call = caller, 
    valid_resopnse = valid_call,
    error_text = error
  )
  
  new_api_call.save()



