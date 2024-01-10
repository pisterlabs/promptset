import os
from openai import OpenAI
from dotenv import load_dotenv
from ai_devs_task import Task
from typing import Dict, Any

load_dotenv()
ai_devs_api_key: str = os.getenv("AI_DEVS_API_KEY", "")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

gnome: Task = Task(ai_devs_api_key, "gnome")
token: str = gnome.auth()
task_content: Dict[str, Any] = gnome.get_content(token)
url: str = task_content["url"]

prompt: str = """
Simply answer the question with just the color name: what is the color of the gnome's hat?
Answer in polish.
If the does not show a gnome with a hat, answer with "ERROR".
"""

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": f"{prompt}"},
        {
          "type": "image_url",
          "image_url": {
            "url": f"{url}",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

answer = response.choices[0].message.content or ""
answer_payload: Dict[str, str] = {"answer": answer}
task_result: Dict[str, Any] = gnome.post_answer(token, answer_payload)
print(task_result)
