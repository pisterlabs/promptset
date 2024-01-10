import openai
import os

openai.api_key = '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'

def generate_text(myPrompt):
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=myPrompt,
      temperature=0.9,
      max_tokens=150,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.6,
      stop=[" Human:", " AI:"]
    )
    message = response.choices[0].text.strip()
    return message

def chat_with_memory(myPrompt):
    # Initialize memory dictionary
    if 'memory' not in globals():
        global memory
        memory = {}
    
    # Add input text to memory
    if 'input' in memory:
        memory['input'].append(myPrompt)
    else:
        memory['input'] = [myPrompt]
    
    # Send input to GPT-3 and get response
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=myPrompt,
      temperature=0.9,
      max_tokens=150,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.6,
      stop=[" Human:", " AI:"]
    )
    response_text = response.choices[0].text.strip()
    
    # Add response to memory
    if 'output' in memory:
        memory['output'].append(response_text)
    else:
        memory['output'] = [response_text]
    
    # Print response
    print(response_text)

while True:
    prompt = input("Enter your prompt: ")
    message = chat_with_memory(prompt)
    print(message)
