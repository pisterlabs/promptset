from utils import *
from config import *
from api_key import api_key, serpapi_api_key

# agents here:
from langchain_core.messages import HumanMessage, SystemMessage
from loader import loader_bill
import requests
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import csv 
import json
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.embeddings import OpenAIEmbeddings

import faiss
from ast import literal_eval
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_experimental.autonomous_agents import BabyAGI
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS



agi_api_embeddings_model = OpenAIEmbeddings()


def styling(image):
  # Initialize the GPT-4 vision model for styling recommendations
  stylist = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)
  print("styling is running")
  # Invoke the styling model with the user's selfie image
  response = stylist.invoke(
    [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "You are now a fashion stylist, based on my selfie (you should call me Master), what styling recommedation should you give to the me (Master) in the picture? \
                      Please make sure to compliment before you judge. start with: 'Master, ...'",
                },
                {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{image}"
          }
        }
            ]
        )
    ]
  )
  
  response = response.content
  print(response)
  return response


def scanner(image):
  scan = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024) # Initialize the GPT-4 vision model for scanning
  print("scanner is running")
  response = scan.invoke(
    [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "What are in the receipt? You are now a receipt scanner that can only out put json, please give me the following columns: 'item_name' and 'quantity'\
                      please make sure to have the real full name (capitalize the first letter) for the item_name, and please only give me the json dictionary without any explainations",
                },
                {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{image}"
          }
        }
            ]
        )
    ]
  ) # Extracting the content from the model's response
  response = response.content
  print(response)
  # Refine and format the response using GPT-3.5
  response = client_helper.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON. give me two columns: 'item_name' and 'quantity', 'for example:{ 'item_name': ['JSPH ORG HUMMUS', 'SB SW BLEND 10Z', 'AMYS LS CHNKY TO', 'AMYS SOUP LS LEN', 'SOUP LENTIL', 'DOLE SPINACH 10'], quantity: [1, 1, 2, 1, 2, 1] }'"},
    {"role": "user", "content": response}
  ]
)
  print(response.choices[0].message.content)
  return response.choices[0].message.content


def voice_grocery(input):
  response = grocery_client.chat.completions.create(
  model="gpt-4 turbo",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON. you need to get two columns: 'item_name' (this need to be the full name of the product) and 'quantity'"},
    {"role": "user", "content": input}
  ]
)
  print(response.choices[0].message.content)
  
  return response.choices[0].message.content


def my_read_csv(csv_file_path):
    """Reads a CSV file and returns its content as a list of dictionaries."""
    try:
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            return list(csv.DictReader(file))
    except IOError as e:
        print(f"Error reading file {csv_file_path}: {e}")
        return None

def update_csv_data(csv_data, new_data, sign = True):
    print(f"CSV data type: {type(csv_data)}")  # Should be a list
    print(f"New data type: {type(new_data)}")  # Should be a list
    def extract_number(quantity_str):
        match = re.search(r'\d+', quantity_str)
        return int(match.group()) if match else 0
    for item in new_data:
        print(f"Item type: {type(item)}")  # Should be a dictionary
        # Make sure item is a dictionary
        if not isinstance(item, dict):
            print(f"Invalid item: {item}")
            continue

        try:
            existing_item = next((row for row in csv_data if row['item_name'] == item['item_name']), None)
            if existing_item:
                existing_quantity = extract_number(existing_item['quantity'])
                item_quantity = extract_number(item['quantity'])
                if sign:
                    new_quantity = existing_quantity + item_quantity
                else:
                    new_quantity = existing_quantity - item_quantity

                # Preserving any unit that might be present in the quantity field
                unit = re.sub(r'\d+', '', existing_item['quantity']).strip()
                existing_item['quantity'] = f"{new_quantity}{unit}"
            else:
                if sign:
                  csv_data.append(item)
        except TypeError as e:
            print(f"Error processing item {item}: {e}")

def write_csv(csv_file_path, csv_data):
    """Writes the updated data to the CSV file."""
    if csv_data:
        try:
            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        except IOError as e:
            print(f"Error writing file {csv_file_path}: {e}")

def input_fridge(grocery_info_json, fridge=fridge):
    csv_file_path = fridge
    csv_data = my_read_csv(csv_file_path)

    if csv_data is not None:
        # Parse the JSON string
        grocery_info = json.loads(grocery_info_json)

        # Transform the JSON structure into a list of dictionaries
        items_list = [{"item_name": name, "quantity": quantity} 
                      for name, quantity in zip(grocery_info["item_name"], grocery_info["quantity"])]

        update_csv_data(csv_data, items_list)
        write_csv(csv_file_path, csv_data)

def update_fridge(grocery_info_json, fridge=fridge):
    csv_file_path = fridge
    csv_data = my_read_csv(csv_file_path)

    if csv_data is not None:
        # Parse the JSON string
        grocery_info = json.loads(grocery_info_json)

        # Transform the JSON structure into a list of dictionaries
        items_list = [{"item_name": name, "quantity": quantity} 
                      for name, quantity in zip(grocery_info["item_name"], grocery_info["quantity"])]

        update_csv_data(csv_data, items_list, sign=False)
        write_csv(csv_file_path, csv_data)


from langchain.llms import OpenAI



def multi_use_api(user_input):
  index = faiss.IndexFlatL2(embedding_size)
  vectorstore = FAISS(agi_api_embeddings_model.embed_query, index, InMemoryDocstore({}), {})

  todo_prompt = PromptTemplate.from_template(
      "You are a planner who is an expert at coming up with a practical todo list (no more than 5 steps) for a given objective. Come up with a informative and full of details todo list (no more than 5 items) for this objective: {objective}. Do not include any items that related to manual operations (paying the fees or purchasing items or tickets or confirm with emails or other needs human verifications, the final step (fifth step) must be giving the final result or the task result I want you to me without any other operation)"
  )
  todo_chain = LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)
  search = SerpAPIWrapper(serpapi_api_key = serpapi_api_key)
  tools = [
      Tool(
          name="Search",
          func=search.run,
          description="useful for when you need to answer questions about current events or situations",
      ),
      Tool(
          name="TODO",
          func=todo_chain.run,
          description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
      ),
  ]


  prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
  suffix = """Question: {task}
  {agent_scratchpad}"""
  prompt = ZeroShotAgent.create_prompt(
      tools,
      prefix=prefix,
      suffix=suffix,
      input_variables=["objective", "task", "context", "agent_scratchpad"],
  )
  llm = OpenAI(temperature=0)
  llm_chain = LLMChain(llm=llm, prompt=prompt)
  tool_names = [tool.name for tool in tools]
  agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
  agent_executor = AgentExecutor.from_agent_and_tools(
      agent=agent, tools=tools, verbose=True
  )

  # Logging of LLMChains
  verbose = False
  # If None, will keep on going forever
  max_iterations: Optional[int] = 5
  baby_agi = BabyAGI.from_llm(
      llm=llm,
      vectorstore=vectorstore,
      task_execution_chain=agent_executor,
      verbose=verbose,
      max_iterations=max_iterations,
  )
  baby_agi({"objective": user_input})
  return baby_agi.final['final']



def get_recipe(user_input):
  def unpack_dicts_to_string(lst_of_dict):
    # Initialize an empty list to store strings
    string_list = []

    # Loop through each dictionary in the list
    for item in lst_of_dict:
        # Unpack each dictionary and format it into a string
        dict_string = ', '.join([f"'{key}': '{value}'" for key, value in item.items()])
        # Append the formatted string to the list
        string_list.append(dict_string)

    # Join all the formatted strings with a separator (e.g., newline)
    result_string = '\n'.join(string_list)
    
    return result_string
  try:
    my_fridge = my_read_csv(fridge)
  except (IOError) as e:
                st.error("Your fridge data is not readable.")
                print(e)
  # Prepare the prompt for recipe generation using available fridge contents
  try: 
    my_fridge_content = unpack_dicts_to_string(my_fridge)
    prompt = "You are a cook and nutritionist can give user(Master) the recipe output JSON. the format should have two columns: 'dish_name' and 'ingredients', and ingredients is a list of ingredients that each contain a dictionary of (item, quantity) for example:{'dish_name': ['Dongpo Meat', 'mac and cheese'], ingredients: [{'pork': 4kg, 'A choy': 2, 'Bell Pepper': 2}, {'cheese': 1tsp, 'egg': 2, 'pasta': 0.5kg}] } You can only use the items from the following" + f": {my_fridge_content} and you should also consider the quantity avaliability of the items"
              
    print(prompt)
    response = cook_bot.chat.completions.create(
      model="gpt-4-1106-preview",
      response_format={ "type": "json_object" },
      messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
      ]
    )
    print(response.choices[0].message.content)
    recipe = response.choices[0].message.content
    # Get detailed recipe and instructions from the cook bot
    response = cook_bot.chat.completions.create(
      model="gpt-4-1106-preview",
      messages=[
        {"role": "system", "content": f"You are a cook and nutritionist can give user(Master) the detail recipe and step by step instructions, you now need to make the dish according to this recipe: {recipe}"},
        {"role": "user", "content": "Please tell me the instruction"}
      ]
    )
    print(response.choices[0].message.content)
    instruction = response.choices[0].message.content
    response = literal_eval(recipe)
  except (IOError) as e:
                st.error("Cook is not available")
                print(e)
  # Update the fridge contents based on the recipe ingredients
  try: 
    for v in response["ingredients"]:
      reply = cook_helper.chat.completions.create(
      model="gpt-3.5-turbo-1106",
      response_format={ "type": "json_object" },
      messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output and extract corresponding JSON. give me two columns: 'item_name' and 'quantity', 'for example:{ 'item_name': ['JSPH ORG HUMMUS', 'SB SW BLEND 10Z', 'Green Pepper', 'Beef Steak', 'SOUP LENTIL', 'DOLE SPINACH 10'], quantity: [1, 1, 2, 1kg, 2, 1] }'"},
        {"role": "user", "content": str(v)}
      ]
    )
    print(reply.choices[0].message.content)
    reply = reply.choices[0].message.content
    update_fridge(reply,fridge=fridge)
  except (IOError) as e:
                st.error("Unable to update the fridge")
                print(e)
  return instruction
