import os
import openai
 
import json
import requests
from typing import List, Optional
from pydantic import BaseModel
 
 

'''
# Setting up Flask and SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///messages.db'
db = SQLAlchemy(app)
'''
 
# import config


# models = config.models
# db = config.db
# uid_context = config.uid_context
# uid = config.uid
# password = config.password
# url = config.url
# username = config.username


class BrowseQuery(BaseModel):
  model: str
  fields: List[str]
  filter: Optional[List[str]]
  limit: Optional[int]
  order: Optional[str]


# Setting up OpenAI
openai.api_key = os.environ['GPT API Keys']


def get_chatgpt_response(phone_number, message):

  # Step 1: send the conversation and available functions to GPT
  messages = [
    {
      "role": "system",
      "content": "Saya asisten yang mengetahui sistem ERP Odoo. Berikan jawaban dengan rinci. Usahakan untuk menyimpulkan jawaban terbaik dari function pada saat function dipanggil"
      #'''Saya adalah Fujicon Boy. Saya merupakan salah satu anggota tim cerdas Fujicon dengan pengetahuan luas di bidang konstruksi, IT, dan multimedia. Saya memiliki kemampuan untuk menganalisis data dari sistem internal yang dibangun dengan platform Odoo, termasuk versi 12 dan versi terbaru.'''
    },
    {
      "role": "user",
      "content": message
    }
  ]


  print(f'\nFirst Message pertama : {messages}')

  if phone_number == '628112227980':
    functions = [
      {
        "name": "browse_odoo",
        "parameters": {
          "type": "object",
          "properties": {
            "model": {
              "type": "string",
              "description": "Model/object that exist in Odoo standard version 12."
            },
            "fields": {
              "type": "array", "items" :{"type" : "string"},
              "description": "fields available in the standard odoo model version 12 plus 'name' or 'display_name' fields, including relevant personals."
            },
            "filter": {
              "type": "array", "items" :{"type" : "string"},
              "description": "filter/domain using Odoo ORM should be in the format '[field,operator,value]'). Use 'ilike' instead of '=' as the operator."
            },
            "order": {
              "type": "string"
            },
            "limit": {
              "type": "integer",
              "description": "Optional. Provide a value of None if not specifically requested."
            }
          },
          "required": ["model", "fields"]
        },
        "description": "Returns data from the specified model.",
      },
    ]
  else:
    functions=[]
  
  response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",               
              messages=messages,
              functions=functions,
              max_tokens=300,
              function_call="auto",
              request_id='chatcmpl-7avQ3v5Fwm9Tmb3AuP1T3vh782lB1',
              user = phone_number
      
  )

  print(f'\nRespon dari OpenAI: {response}')
  # Get the assistant's reply
  response_message = response["choices"][0]["message"]
  response_message_content =  response["choices"][0]["message"]['content']
  token = response['usage']['total_tokens']

  print(f'\nTotal Tokens: {token}')

  # Step 2: check if GPT wanted to call a function >> PNGIRIMAN KEDUA
  if response_message.get("function_call"):
    # Step 3: call the function
    # Note: the JSON response may not always be valid; be sure to handle errors
    available_functions = {"browse_odoo": browse_odoo}
    function_name = response_message["function_call"]["name"]
    fuction_to_call = available_functions[function_name]

    print(f'''\n\nPengiriman ke dua. \nfunction_name : {function_name}
              ''')
        
    function_args = json.loads(response_message["function_call"]["arguments"])
    #print(f'\n\nArgument untuk memanggil Function :\n{function_args}')
    
    data = browse_odoo(function_args)
    function_response = json.dumps(data)
    

    # Step 4: send the info on the function call and function response to GPT
    messages.append(
      response_message)  # extend conversation with assistant's reply
    messages.append({
      "role": "function",
      "name": function_name,
      "content": function_response,
    })  # extend conversation with function response

    print(f'\nSecond Message = {messages}')

    # get the second response from GPT where it can see the function response
    second_response = openai.ChatCompletion.create(
                      model="gpt-3.5-turbo",
                      messages=messages,
                      max_tokens=1000,
                      user = phone_number,
                      # echo = True
                      
    ) 

    second_response_content =  second_response["choices"][0]["message"]['content']
    second_token = second_response['usage']['total_tokens']

    
    print(f'\n\nRespon dari OpenAI yang kedua:\n {second_response}')
    return second_response_content + f"\n\n({token}+{second_token}={token+second_token})\n\n{function_args}\n\n{function_response}"

  # Mode debug
  response_str = str(response)
  #response_str = json.dumps(response.json(), indent=2)

  if response_str is None:
    response_str = ""
  response_str += "\n\n\n[DEBUG] Response from API:\n" + response_str + "\n\n----------\n\n" + str(
    messages)

  # Save the user message and bot response to the database
  user_msg = Message(phone_number=phone_number,
                     message=message,
                     result=response_str,
                     incoming=True)
  
  db_sqlalchemy.session.add(user_msg)
  db_sqlalchemy.session.commit()

  return response_message_content + f"\n\n({token})"




#===================================================

def browse_odoo(function_args):
  print('\nMasuk ke proses GET_BROWSE :')

  model   = function_args['model']
  fields  = ['display_name','name']
  fields_to_add = function_args['fields']
  for field in fields_to_add:
    if field not in fields:
        fields.append(field)
   
   
  filter  = [function_args.get('filter',[])]
  print(f'Filter sebelum diolah : {filter} , lenght : {len(filter)}')

  try:
    if len(filter[0]) > 0:
      if filter[0][1] == "=" and isinstance(filter[0][2], str):
        filter[0][1] = "ilike"
      filter2 = [filter]
      filter = [filter]
        #filter = [filter]
      print(f'Filter 2 : {filter2} ')
    print(f'Filter setelah diolah : {filter} ')

  except Exception as e:
     print(f'Error (filter tidak diolah) : {e}')
     
    
    
  order   = function_args.get('order', None)
  limit   = function_args.get('limit', 30)

  #Adujstmen untuk filter, fields, limit dan order.
  #if filter is not None:
    #filter = ast.literal_eval(filter)
  #else:
   # filter = []
  
  params = {'fields': fields}
  if limit is not None:
    params['limit'] = limit
  if order is not None:
    params['order'] = order

  print(f'\n\nParameters : {params}')
  print(f'Filter : {filter}')

  # data = models.execute_kw(db, uid, password, model, 'search_read', filter, params)

  
  run_str = f"models.execute_kw('{str(db)}', '{str(uid)}', '{str(password)}', '{str(model)}', 'search_read', {str(filter)}, {str(params)})"
  
  print(f'Command : {run_str}')

  try:
    data = eval(run_str)
  
    # data = models.execute_kw(db, uid, password, model, 'search_read', filter, params)
  except Exception as e:
    print(e)
    error_str = '{"error":' + str(e) + '}'
    # Handle the error here
    data = json.loads(error_str)


  return data


 
