import openai
import os
from dotenv import load_dotenv

# set the API key
load_dotenv()
openai.api_key =  os.getenv("OPENAI_KEY")

# set the model to use (e.g. "text-davinci-002")
model = "gpt-3.5-turbo"

def ask_chatGPT(prompt):
    message = {"role": "user", "content": prompt}

    # generate text
    completions = openai.ChatCompletion.create(
        model=model,
        messages=[message],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=1,
    )

    # print the generated text
    return completions.choices[0]["message"]["content"]
   
   
   
if __name__ == "__main__":
        
    prompt = ""
    while prompt != "quit()":

        # set the prompt
        prompt = input("\nAsk me anything. Type quit() to terminate: ")
        
        print(ask_chatGPT(prompt))
        
