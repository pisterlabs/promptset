import os
from openai import OpenAI
from dotenv import load_dotenv
from ai_devs_task import Task
from typing import Dict, Any
from helpers import send_request


load_dotenv()
ai_devs_api_key: str = os.getenv("AI_DEVS_API_KEY", "")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

scraper: Task = Task(ai_devs_api_key, "scraper")

token: str = scraper.auth()
task_content: Dict[str, Any] = scraper.get_content(token)

url: str = task_content["input"]
question: str = task_content["question"]

response_text: str = send_request("GET", url)

system: str = f"""
You answer the question concisely, in one sentence.
Answer using the following knowledge:
{response_text}
"""

answer = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ]
    )

task_answer: str = answer.choices[0].message.content or ""
answer_payload: Dict[str, str] = {"answer": task_answer}
task_result: Dict[str, Any] = scraper.post_answer(token, answer_payload)
print(task_result)
