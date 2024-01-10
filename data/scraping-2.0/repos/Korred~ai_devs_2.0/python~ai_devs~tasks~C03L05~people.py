import os

import openai
from dotenv import load_dotenv
from icecream import ic
from utils.client import AIDevsClient
import httpx

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Get API key from environment variables
aidevs_api_key = os.environ.get("AIDEVS_API_KEY")

# Create a client instance
client = AIDevsClient(aidevs_api_key)

# Get a task
task = client.get_task("people")
ic(task.data)

# Extract question
question = task.data["question"]

# Extract the name of the person from the question (reverse diminutive form)
system_msg = """
Extract the name and surname of the person from the question provided to you.
Ensure to transform the name into its full form / non-diminutive form e.g.
"Krzysiek" -> "Krzysztof"
"Tomek" -> "Tomasz"
"Jarek" -> "Jarosław"
"Kasia" -> "Katarzyna"

Return the name and surname in the following format: "Name Surname"
"""
extracted_name = (
    openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ],
        max_tokens=100,
    )
    .choices[0]
    .message.content
)

ic(extracted_name)


# Load the list of names and information about then
response = httpx.get(task.data["data"])

# Create a dictionary of names
names = {f"{entry['imie']} {entry['nazwisko']}": entry for entry in response.json()}

person = names[extracted_name]
system_msg = f"""
Use the following facts about the person to answer the questions provided to you:
Name: {person['imie']}
Surname: {person['nazwisko']}

General information: {person['o_mnie']}
Age: {person['wiek']}
Favourite Kapitan Bomba character: {person['ulubiona_postac_z_kapitana_bomby']}
Favourite TV series: {person['ulubiony_serial']}
Favourite movie: {person['ulubiony_film']}
Favourite colour: {person['ulubiony_kolor']}

Answer in Polish.
"""

answer = (
    openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ],
        max_tokens=200,
    )
    .choices[0]
    .message.content
)

ic(answer)

response = client.post_answer(task, answer)
ic(response)
