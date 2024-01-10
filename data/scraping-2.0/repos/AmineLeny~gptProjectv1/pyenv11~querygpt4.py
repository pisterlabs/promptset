

import os
from fastapi import FastAPI, HTTPException
from openai import OpenAI
import csv
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
messages = []
class ChatGPTMicroservice:
    
 
    def __init__(self, api_key, data_dir="/app/data"):
        self.openai_handler = OpenAI(api_key=api_key)
        self.csv_file = os.path.join(data_dir, "qa_data.csv")  

      

    def query_chatgpt_api(self, question, model, max_tokens, temperature):
        messages.append({"role": "user", "content": question})
        
        response = self.openai_handler.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        generated_response = response.choices[0].message.content
        messages.append({"role":"assistant","content":generated_response})
        
        self.save_to_csv(question, generated_response)
        return generated_response

    def save_to_csv(self, question, answer):
        with open(self.csv_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["question", "answer"])
            writer.writerow({"question": question, "answer": answer})

@app.post("/ask")
async def ask(question: str, model: str = "gpt-3.5-turbo", max_tokens: int = 150, temperature: float = 0.7):
    """Ask a question using the ChatGPT microservice."""
    try:
        microservice = ChatGPTMicroservice(api_key=os.getenv("OPENAI_API_KEY"))
        reponse = microservice.query_chatgpt_api(question, model, max_tokens, temperature)
        return {"GPT": reponse }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    