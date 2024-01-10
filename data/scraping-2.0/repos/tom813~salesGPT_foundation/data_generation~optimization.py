#!/usr/bin/env python
# coding: utf-8

# In[14]:


import openai
import random
import json

# Set OpenAI API Key
openai.api_key = ""

def get_openai_completion(messages):
    """
    Calls OpenAI API to get model's response based on provided messages.
    
    Parameters:
    messages: List of dictionaries containing role and content information.
    
    Returns:
    String: Content of the model's response.
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.8,
        messages=messages
    )
    
    return completion.choices[0].message.content

def evaluate_and_generate_conversations(conversation, num_samples=1):
    """
    Evaluates conversations and generates new ones based on suggestions.
    
    Parameters:
    conversation: List of strings representing a conversation.
    num_samples: Integer representing the number of samples to generate.
    
    Returns:
    List: A list containing generated conversations based on the suggestions.
    """
    successful = "<EndOfConversation>" in conversation
    
    generated_conversations = []
    if not successful:
        suggestion_list = [f"#suggestion {i}" for i in range(1, num_samples + 1)]
        suggestions_prompt = f'''
        Give me {num_samples} suggestions on what the sales agent should improve from a psychological perspective 
        when it comes to selling a product based on this conversation:
        {conversation}
        
        return only a python list of the suggestions like the following using double quotes:
        {suggestion_list}
        '''
        suggestions_messages = [{"role": "user", "content": suggestions_prompt}]
        suggestions = json.loads(get_openai_completion(suggestions_messages))
        
        for suggestion in suggestions:
            generation_prompt = f'''
            Here is a conversation and a suggestion to that conversation. 
            conversation: {conversation}
            
            suggestion: {suggestion}
            
            Generate a short example conversation based on the suggestion and related to the conversation.
            
            The output should be parsable by json.loads() and the json should be in double quotes
            return the modified conversation in the same form as the input conversation:
            ["Customer: text of customer", "Salesman: text of salesman", "Customer: text of customer", etc.]
            '''
            generation_messages = [{"role": "user", "content": generation_prompt}]
            new_conversation = json.loads(get_openai_completion(generation_messages))
            print("##########################")
            print(new_conversation)
            generated_conversations.append(new_conversation)
    
    return generated_conversations

def append_to_json_file(data, filename):
    """
    Appends data to a JSON file or creates one if it doesn't exist.
    
    Parameters:
    data: Data to append.
    filename: Name of the file.
    """
    try:
        with open(filename, "r") as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_data.extend(data)
    
    with open(filename, "w") as file:
        json.dump(existing_data, file, indent=4)


success_optimization = []
defeat_optimization = []

# Load conversation data
with open("sample_conversations.json", "r") as file:
    raw_data = json.load(file)
    
    cleaned_data = [x for x in raw_data if x != "Customer: " and x != "<Defeat>"]
    
    for conversation in cleaned_data:
        try: 
            if "<Success>" in conversation:
                optimizations = evaluate_and_generate_conversations(conversation, 1)
                success_optimization.extend(optimizations)
            elif "<Defeat>" in conversation:
                optimizations = evaluate_and_generate_conversations(conversation, 3)
                defeat_optimization.extend([o for o in optimizations if "Customer: " in o])
        except Exception as e:
            print(e)

# Append generated conversations to respective files
append_to_json_file(defeat_optimization, "defeat_optimization.json")
append_to_json_file(success_optimization, "success_optimization.json")


# In[42]:


import pandas as pd
from datasets import load_dataset

# Loading dataset from Huggingface and converting to Pandas DataFrame
ds = load_dataset("goendalf666/sales-conversations-instruction", split="train")
df = ds.to_pandas()

# Initialize a list to store modified conversations
modified_conversations = []
joined_conversations = defeat_optimization + success_optimization

# Navigate through dataframe and create prompts
for c in joined_conversations:
    current_conversation = ""
    
    try:
        for statement in c:
            if "Customer:" in statement:
                current_conversation += statement + " "
            elif "Salesman:" in statement:
                prompt = f"""You are in the role of a Salesman. Here is a conversation:
                {current_conversation}
                
                Answer as a Salesman to the previous Statement to convince the person to buy the product or service.
                {statement}
                """
                modified_conversations.append(prompt)
                current_conversation += statement + " "
    except Exception as e:
        print(e)
        


# In[43]:


df_dict = df.to_dict(orient='list')

for i in range(len(modified_conversations)):
    df_dict["0"].append(modified_conversations[i])
    


# In[45]:


df = pd.DataFrame(df_dict)
ds = Dataset.from_pandas(df)


# In[52]:


ds.push_to_hub("goendalf666/sales-conversations-instruction-ext")


# In[ ]:




