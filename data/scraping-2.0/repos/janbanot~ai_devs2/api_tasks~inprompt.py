import os
from openai import OpenAI
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
from ai_devs_task import Task

load_dotenv()
ai_devs_api_key: str = os.getenv("AI_DEVS_API_KEY", "")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

inprompt: Task = Task(ai_devs_api_key, "inprompt")
token: str = inprompt.auth()
task_content: Dict[str, Any] = inprompt.get_content(token)

knowledge_dict: Dict[str, str] = {}
input: List[str] = task_content["input"]
for entry in input:
    words: List[str] = entry.split()
    name: str = words[0]
    knowledge_dict[name] = entry

question: str = task_content["question"]
name_pattern: str = r"\b[A-Z][a-z]*\b"
subject: str = re.findall(name_pattern, question)[0]
subject_info: str = knowledge_dict[subject]

prompt: str = f"""
Answer the question shortly using only the information given below:
{subject_info}
"""

model_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
)
task_answer: str = model_response.choices[0].message.content or ""

answer_payload: Dict[str, str] = {"answer": task_answer}
task_result: Dict[str, Any] = inprompt.post_answer(token, answer_payload)
print(task_result)
