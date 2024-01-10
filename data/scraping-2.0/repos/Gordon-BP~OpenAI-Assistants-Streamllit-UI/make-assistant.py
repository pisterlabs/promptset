import openai
import os, json
from dotenv import load_dotenv, set_key, find_dotenv
from util.logger import logger

dotenv_path = find_dotenv(usecwd=True)
if not dotenv_path:
    dotenv_path = '.env'  # Set a default path if not found
load_dotenv()

# Set new values or modify existing ones
client = openai.Client(api_key = os.environ['OPENAI_API_KEY'])
name = (name := os.environ.get('OPENAI_ASSISTANT_NAME')) or "My Bot"
if "ASSISTANT_INSTRUCTIONS" in os.environ:
    instructions = os.environ["ASSISTANT_INSTRUCTIONS"]
else:
    # If not, read the contents of the file into a single string
    with open("instructions.txt", "r") as file:
        instructions = file.read()
with open("oai_tools.json", "r") as file:
    data = json.load(file)
# Convert the Python object into a JSON string
json_string = json.dumps(data)
assistant = client.beta.assistants.create(
  name=name,
  description="Let's have some fun! You will generate short, zany quizzes in the style of mid 2010's buzzfeed quizzicles",
  instructions = instructions,
  model="gpt-4-1106-preview",
  tools=[{"type": "function", "function":func} for func in data],
)

set_key(dotenv_path, 'OPENAI_ASSISTANT_ID', assistant.id) 