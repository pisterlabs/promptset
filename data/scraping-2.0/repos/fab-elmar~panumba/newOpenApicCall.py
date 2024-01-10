import json
import os
from datetime import datetime

from openai import OpenAI

key=os.environ.get("OPENAI_API_KEY")

client = OpenAI()
MODEL = "gpt-3.5-turbo"
# INSTRUCTION = "Provide an answer in two parts the answer should have to key value paiars. The first key 'number' should have an integer as value that is logically associated with the question or in case there is the direct answer. The second part, under the key 'context', should be a brief explanation of the logic used to derive the number. For direct questions like 'How many days are in a week?', return the factual answer. For abstract questions like 'What color is the sky?', derive a number using a logical method and explain this method in the context." 
modifyed_instruction = "Provide an answer in two parts: the answer should have two key-value pairs both enclosed in double quotes pairs. The format must be parseable as json. The first key 'number' must be an integer written out in full with all trailing zeros (e.g., 4,500,000 for 4.5 million) as the value that is logically associated with the question or, in case there is a direct answer. The second part, under the key 'context', should be a brief explanation of the logic used to derive the integer number. For direct questions like 'How many days are in a week?', return the factual answer as an integer. For abstract questions like 'What is the age of the sea?', derive an integer using a logical method and explain this method in the context."

responses = []

def print_green(text):
    print("\033[92m" + text + "\033[0m")

def print_red(text):
    print("\033[91m" + text + "\033[0m")
    
    
    
    

    
while True:
    
    user_question = input("Enter your question (or 'q' to quit): ")

    if user_question.lower() == 'q':
        print_red("Goodbye, 42 be with you!") 
        break
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": modifyed_instruction  },
            {"role": "user", "content": user_question},
        
        ],
        temperature=0,
    )
    
    
    
  
    print(response)
    content = response.choices[0].message.content
    content = content.replace("'", "\\'").replace("\\", "\\\\")
    
    try:
        response_content = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON content: {e}")
        print_red(content)
        continue
     
    #  response_content = json.loads(content)

    # Extract 'number' and 'context' values
    number = response_content.get('number')
    context = response_content.get('context')
    
    print_red(f"Number: {number}")
    print_green(f"Context: {context}")
    
    response_data = {
        "question": user_question,
        "number": number,
        "context": context
    }
    
    responses.append(response_data)



# Save responses to file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

filename = f"responses_{timestamp}.json"
with open(filename, 'w') as json_file:
    json.dump({i+1: response for i, response in enumerate(responses)}, json_file, indent=4)

print(f"Responses saved to {filename}")