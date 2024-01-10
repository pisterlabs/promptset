import os
import openai
import requests  # type: ignore
import tempfile
from dotenv import load_dotenv
from ai_devs_task import Task
from typing import Dict, Any

load_dotenv()
ai_devs_api_key: str = os.getenv("AI_DEVS_API_KEY", "")
openai.api_key = os.getenv("OPENAI_API_KEY", "")

whisper: Task = Task(ai_devs_api_key, "whisper")
token: str = whisper.auth()

content: Dict[str, Any] = whisper.get_content(token)
file_url: str = "https://zadania.aidevs.pl/data/mateusz.mp3"

response: requests.Response = requests.get(file_url)
if response.status_code == 200:
    # Create a temporary file to save the MP3 content
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        # Write the MP3 content to the temporary file
        temp_file.write(response.content)
        temp_file.flush()
        temp_file.seek(0)  # Reset file pointer to the beginning

    with open(temp_file.name, "rb") as audio_file:
        transcript: str = openai.Audio.transcribe("whisper-1", audio_file)["text"]
        answer_payload: Dict[str, str] = {"answer": transcript}
        task_result: Dict[str, Any] = whisper.post_answer(token, answer_payload)
        print(task_result)
else:
    print("Failed to download the MP3 file")
