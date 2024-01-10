from openai import OpenAI
import subprocess, os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

def index():
    with open('../Generation/data/concept_input.txt', 'r') as f:
        content = f.read()
    print("Generating script...")
    summary_prompt = f"Write a audio script explainaing the concept of {content}. Explan it like a 5 year old. Explain it in detail. Everything in 1 paragraph. Max 250 words.  Start with the script. No text before it. Avoid starting with - Sure, ..... Start directly - <concept>....."
    
    response = client.chat.completions.create(
      messages=[
            {"role": "user", "content": summary_prompt}
        ],
      model="gpt-4",
    )

    with open("../Generation/data/audio.txt", "w") as f:
        f.write(response.choices[0].message.content)

    #subprocess.call(["python3", "gif_gen.py"])
    
index()
