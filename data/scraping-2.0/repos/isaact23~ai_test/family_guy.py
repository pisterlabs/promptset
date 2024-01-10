from openai import OpenAI

client = OpenAI()

def run_conversation():
    # Step 1: send the conversation and available functions to GPT
    messages = [{"role": "user", "content": "Generate an alternative ending script for the Spongebob episode when they deliver the pizza but forget the drink."}]
    response = client.chat.completions.create(model="gpt-3.5-turbo-0613",
    messages=messages)
    response_message = response["choices"][0]["message"]["content"]
    return response_message

print(run_conversation())