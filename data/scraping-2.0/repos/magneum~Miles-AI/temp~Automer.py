import os
import json
import time
import openai
import random
from Agents import Agents
from dotenv import load_dotenv
from colorama import Fore, Style

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY", "")
objective = "Create a Python code that implements a vector database using numpy for the BabyAGI system, enabling efficient storage and retrieval of vector representations for data processing and inference. Refactor the existing implementation by replacing the usage of Pinecone with numpy for the vector database to improve data processing and inference capabilities. Provide guidance on how to implement this custom Python code using numpy for the vector database in the BabyAGI system, and ensure efficient storage and retrieval of vector representations for data processing and inference."
num_iterations = 4
max_tokens = 2048
wait_time = 30


def generate_response(agent, objective):
    agent_name = agent["name"]
    prompt = f"As the {agent_name}, my goal is to {agent['motive']}.\n\nTask: {agent_name}: {agent['prompt']}{objective}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=random.uniform(0.5, 1),
    )
    return response.choices[0].text.strip()


def ada_embedding(text):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    return embedding


data = []
for i in range(num_iterations):
    print(f"Iteration {i+1} of agent communication")
    for agent in Agents:
        try:
            response_text = generate_response(agent, objective)
            response_embedding = ada_embedding(response_text)
        except Exception as e:
            response_text = str(e)
            response_embedding = None
        agent_name = agent["name"]
        print(f"{Fore.CYAN}{agent_name}:{Style.RESET_ALL} {response_text}")
        data.append(
            {
                "agent": agent_name,
                "text": response_text,
                "embedding": response_embedding,
            }
        )
        time.sleep(random.uniform(0.5, 1.5))
    if i < num_iterations - 1:
        print(f"Waiting for {wait_time} seconds before next iteration...")
        time.sleep(wait_time)

with open(f"Automer.json", "w") as f:
    json.dump(data, f)
print(f"Agent responses have been generated and saved to Automer.json file.")

with open("final.txt", "w") as f:
    for agent in data:
        f.write(f"{agent['agent']}: {agent['text']}\n")
print("Final output has been saved to final.txt file.")
