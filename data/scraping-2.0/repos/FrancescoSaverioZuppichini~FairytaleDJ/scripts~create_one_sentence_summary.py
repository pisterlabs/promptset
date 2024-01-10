"""
This script takes all the songs we have and create a summary for each lyric
"""

from dotenv import load_dotenv

load_dotenv()

import json
from collections import defaultdict
from pathlib import Path
from pprint import pprint

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["song"],
    template=Path("prompts/summary.prompt").read_text(),
)

llm = ChatOpenAI(temperature=0)

chain = LLMChain(llm=llm, prompt=prompt)

with open(
    "data/lyrics_with_spotify_url.json",
    "r",
) as f:
    data = json.load(f)

lyrics_summaries = {}

for movie, lyrics in data.items():
    for lyric in lyrics:
        print(f"Creating summary for {lyric['name']}")
        summary = chain.run(song=lyric["text"])
        lyrics_summaries[lyric["name"].lower()] = {
            "summary": summary,
            "embed_url": lyric["embed_url"],
        }

with open(
    "data/lyrics_with_spotify_url_and_summary.json",
    "w",
) as f:
    json.dump(lyrics_summaries, f)

pprint(lyrics_summaries)
