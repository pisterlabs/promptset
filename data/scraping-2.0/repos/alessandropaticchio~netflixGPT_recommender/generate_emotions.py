from collections import defaultdict

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from pathlib import Path
import pandas as pd

prompt = PromptTemplate(input_variables=["title", "summary", "genres"],
                        template=Path("prompts/item_emotions.prompt").read_text())

llm = ChatOpenAI(temperature=0.7)

chain = LLMChain(llm=llm, prompt=prompt)

df = pd.read_csv("data/titles.csv").sample(n=2000)

new_data = []

def parse_output(res):
    try:
        emotions, emojis = res.split('\n')
        emotions = emotions.split('emotions: ')[1]
        emojis = emojis.split('emojis: ')[1]
    except:
        print("Error validating the output...")
        raise ValueError
    return emotions, emojis


for i, row in df.iterrows():
    res = chain.run(title=row["title"], summary=row["description"], genres=row["genres"])
    try:
        emotions, emojis = parse_output(res)
    except ValueError:
        continue
    new_data.append([row["title"], row["description"], row["genres"], emotions, emojis])

new_df = pd.DataFrame.from_records(new_data, columns=["title", "description", "genres", "emotions", "emojis"])

new_df.to_csv("data/titles_with_emotions_emojis.csv", index=False)
