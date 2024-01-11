from openai import OpenAI

client = OpenAI()

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model,
    messages=messages,
    temperature=0)
    return response.choices[0].message["content"]
    
    
    
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = client.chat.completions.create(model=model,
    messages=messages,
    temperature=temperature)
#     print(str(response.choices[0].message))
    return response.choices[0].message["content"]