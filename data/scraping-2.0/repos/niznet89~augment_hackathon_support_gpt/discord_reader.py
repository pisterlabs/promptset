from llama_index import DiscordReader
from llama_index import download_loader
import os
import nest_asyncio
nest_asyncio.apply()
from llama_index import ServiceContext
import openai
import re
import csv
import time
import random
from dotenv import load_dotenv
import os
from llama_index import Document

load_dotenv()


openai_api_key = os.environ.get("OPENAI_API")
discord_key = os.environ.get("DISCORD_TOKEN")

os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = openai_api_key

def hit_discord():

    DiscordReader = download_loader('DiscordReader')
    discord_token = discord_key
    channel_ids = [1088751449271447552]  # Replace with your channel_i

    #channel_ids = [1057178784895348746]  # Replace with your channel_id
    reader = DiscordReader(discord_token=discord_token)
    documents = reader.load_data(channel_ids=channel_ids)
    print("docs length", len(documents))
    #discord_token = os.getenv("MTA4MjQyOTk4NTQ5Njc3MjYyOA.G8r0S7.MURmKr2iUaZf6AbDot5E_Gad_10oGbrMFxFVy4")

    #documents = DiscordReader(discord_token="MTA4MjQyOTk4NTQ5Njc3MjYyOA.G8r0S7.MURmKr2iUaZf6AbDot5E_Gad_10oGbrMFxFVy4").load_data(channel_ids=channel_ids, limit=[10])
    service_context = ServiceContext.from_defaults(chunk_size_limit=3000)
    nodes = service_context.node_parser.get_nodes_from_documents(documents)

    print("nodes length:", len(nodes))

    questions = {}
    array_of_docs = []
    for n in nodes:
        print(n)
        prompt = f"""You are tasked with parsing out only the text from Discord messages (including who wrote it and their role). Here is the Discord data: {n}"""

        MAX_RETRIES = 3
        SLEEP_TIME = 0.75  # in seconds

        for _ in range(MAX_RETRIES):
            try:
                time.sleep(round(random.uniform(0, SLEEP_TIME), 2))
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                        messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                break  # If the API call works leave loop
            except Exception as e:

                print(f"Error calling OpenAI API: {e}")
                time.sleep(SLEEP_TIME)


        #print(completion.choices[0].message['content'])
        text = completion.choices[0].message['content']
        document = Document(text=text)
        array_of_docs.append(document)
    print(array_of_docs)
    return array_of_docs


__all__ = ['hit_discord']
