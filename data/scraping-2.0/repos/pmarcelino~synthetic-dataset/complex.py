import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
target_user = "End User"


def get_llm_response(user_prompt, system_prompt, model="gpt-3.5-turbo", temperature=0):
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def write_to_file(content, file_path):
    with open(file_path, "w") as f:
        f.write(content)


# Read CSV
df = pd.read_csv("mendable_docs_data.csv")

df = df.iloc[
    [33]
]  # Restrict data size by specifying a row (row number = row number in Excel - 2)

# Create complex questions
system_prompt = """
You are an expert assistant who can help creating questions that require functional operations within a knowledge graph context. 

These questions should be designed to challenge the understanding of how functional operations—such as aggregation, comparison, filtering, sorting, pathfinding, temporal operations, and logical reasoning—are applied to extract and process complex information from a knowledge graph. 

Please provide questions that cover a variety of scenarios where multiple data manipulations are needed to obtain an answer. Each question should clearly indicate the functional operations involved and the conceptual understanding necessary to solve it. 
"""

questions = []
for content in df["content"]:
    write_to_file(content, "mendable_content_used.txt")

    user_prompt = f"Your task is to create questions, from the perspective of a **{target_user}**, based on the information below.\n\n---\n\n{content}"
    response = get_llm_response(user_prompt, system_prompt, model="gpt-4-1106-preview")
    questions.append(response)

    write_to_file(response, "mendable_complex_questions.txt")
