import openai
from dotenv import load_dotenv
import os
model = {3:"gpt-3.5-turbo", 4:"gpt-4-1106-preview"}
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_KEY"))

def get_prompts(file="prompt.txt"):
    with open(file, "r") as f:
        prompts = f.read().split("\n\n")
        return[prompt.replace("\n", "") for prompt in prompts]
    
def generate_questions(user_prompt):
    prompts = get_prompts()
    messages =[{"role": "system","content": prompts[0]},
    {"role": "user","content": prompts[1]},
    {"role":"assistant", "content":prompts[2]},
    {"role": "user","content": f"{user_prompt}"},]
    return client.chat.completions.create(
        model=model[3],
        messages = messages,
        n=1,
    ).choices[0].message.content

def main():
    user_input = input("Text: ")
    print(generate_questions(user_prompt = user_input))

if __name__ == "__main__":
    main()