import pandas as pd
from langchain.prompts import ChatPromptTemplate
import json

def parse_csv_to_dict(filename):
    df = pd.read_csv(filename, encoding='utf-8')
    return df.to_dict(orient='records')

def lc_to_oai_role(role):
  returned_role = ''
  if role == 'ai':
    returned_role = 'assistant'
  if role == 'system':
    returned_role = 'system'
  if role == 'human':
    returned_role = 'user'

  return returned_role

def get_message_representation(messages):
    message_representation = []
    for message in messages:
        role = lc_to_oai_role(message.type)
        message_dict = {"role": role, "content": message.content}
        message_representation.append(message_dict)
    return message_representation

def write_messages_to_local(input_file, output_file):
  # Overwrites existing file or creates it if it doesn't exist. 
  with open(output_file, 'w', encoding='utf-8'):
    pass
 
  data_list = parse_csv_to_dict(input_file)

  messages = [
    ("system", SYSTEM_MESSAGE),
    ("user", "{input}"),
    ("assistant", "{output}")
  ]

  message_prompt = ChatPromptTemplate.from_messages(messages)
  
  for index, _ in enumerate(data_list):
      with open(output_file, 'a', encoding='utf-8') as FT:
        message_list = message_prompt.format_messages(**data_list[index])
        formatted_message = get_message_representation(message_list)
        messages_to_save = {"messages": formatted_message }
        FT.write(json.dumps(messages_to_save, ensure_ascii=False) + '\n')


SYSTEM_MESSAGE = "" # you are a helpful assistant
INPUT_FILE = ''  #  training_data.csv
OUTPUT_FILE = '' # formatted_messages.jsonl
write_messages_to_local(INPUT_FILE, OUTPUT_FILE)
