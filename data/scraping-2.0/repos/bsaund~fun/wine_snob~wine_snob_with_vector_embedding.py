"""
This is a quick look at using vector embeddings to match input data with queries.
To use this, you will set to set your environment variable for the openAI key
https://github.com/openai/openai-python#usage
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from openai.embeddings_utils import get_embedding
import openai
# import llm_extraction
import pickle
import logging

ENTRIES = [
    "The wine from the party in winter of 2016 had a picture of a dragon on the label. Eric said it was expensive.",
    "This wine was from 2018 and had a green label. It was okay, but Ive had better",
    "The merlot from the San Carlos diner is really great. It is a Chateau de Saund 2017. It paired really nicely with the fish",
    "The pinot noir from the San Carlos diner is awful. It is a Chateu de Cochran 1956. Never get this again",
    "Donna has put on a wonderful garden party and is serving a bunch of great wines. I am trying the Chateau Clairac Blaye Cotes de Bordeau, 2018",
    "At Donna\'s party, she served a 2017 Chateu Haut-Pezat that I thought was excellent with salmon. I give it 91",
]

QUERIES = [
    "I want the wine with the dragon label",
    "What was the wine I had from Chateau de Saund?",
    "I went to the San Carlos diner and had a great wine. What was it?",
    "Laura said some wine was really expensive. Or maybe it was Eric.",
    "I had a terrible wine at the San Carlos diner from some Chateu. What was it?",
    "I'm at the store and I see a 2018 Chateau Claric Bordeaux.  I'm having salmon tonight. Is this one any good?",
]


def construct_ws_dataframe():
    df = pd.DataFrame(columns=["text", "wine", "wine_embedding", "vector_embedding"])
    return df


class WineSnob:
    def __init__(self):
        self.saved_queries_file = "./saved_queries.pkl"
        self.saved_queries = dict()
        self.load_saved_queries()
        self.df = self.wip_add_to_df(construct_ws_dataframe())

    def __del__(self):
        self.save_saved_queries()

    def load_saved_queries(self):
        try:
            with open(self.saved_queries_file, "rb") as f:
                self.saved_queries = pickle.load(f)
        except FileNotFoundError:
            logging.warning("No saved queries found")

    def save_saved_queries(self):
        with open(self.saved_queries_file, "wb") as f:
            pickle.dump(self.saved_queries, f)

    def extract_wine_from_entry(self, entry_text):
        prompt = f"""Each examples references a wine. What is the wine?
        Example: The 2017 Chateau de ste Michelle Chardonnay wine was excellent. I had it at a garrison keilor concert in Washington state. I believe I had the wine at the winery. It was about 2013.
        Answer: Chardonnay, 2017, Chateau de ste Michelle
    
        Example: The pinot noir from the San Carlos diner is awful. It is a Chateau de Cochran 1956. Never get this again
        Answer: Pinot Noir, 1956, Chateau de Cochran
    
        Example: Donna has put on a wonderful garden party and is serving a bunch of great wines. I am trying the Chateau Clairac Blaye Cotes de Bordeau, 2018
        Answer: Bordeau, 2018, Chateau Clairac Blaye Cotes
    
        Example: At Donna\'s party, she served a 2017 Chateu Haut-Pezat that I thought was excellent with salmon. I give it 91
        Answer: [Unknown], 2017, Chateau Haut-Pezat
    
        Example: {entry_text}
        Answer: 
        """
        # print(prompt)
        if prompt in self.saved_queries:
            return self.saved_queries[prompt]

        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        # print(completion)
        wine = completion["choices"][0]["message"]["content"]
        print(wine)
        self.saved_queries[prompt] = wine
        return wine

    def extract_wine_from_query(self, entry_text):
        prompt = f"""Each examples references a wine. What is the wine?
        Example: Have I had the 2017 Chateau de ste Michelle Chardonnay before?
        Answer: Chardonnay, 2017, Chateau de ste Michelle
    
        Example: I'm looking at a pinot noir. It is a Chateau de Cochran 1956. Have I had this?
        Answer: Pinot Noir, 1956, Chateau de Cochran
    
        Example: I think I've had the Chateau Clairac Blaye Cotes de Bordeau from 2018 before. Tell me about it.
        Answer: Bordeau, 2018, Chateau Clairac Blaye Cotes
    
        Example: I see this 2017 Chateu Haut-Pezat. Have I rated it?
        Answer: [Unknown], 2017, Chateau Haut-Pezat
    
        Example: {entry_text}
        Answer: 
        """
        # print(prompt)
        if prompt in self.saved_queries:
            return self.saved_queries[prompt]
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        # print(completion)
        wine = completion["choices"][0]["message"]["content"]
        print(wine)
        self.saved_queries[prompt] = wine
        return wine

    def embed(self, text):
        key = "embedding: " + text
        if key not in self.saved_queries:
            embedding_model = "text-embedding-ada-002"
            embedding = get_embedding(text, engine=embedding_model)
            self.saved_queries[key] = embedding
        return self.saved_queries[key]




    def insert_embedding_to_def(self, df, text):
        wine = self.extract_wine_from_entry(text)
        df.loc[len(df)] = {"text": text, "wine": wine, "wine_embedding": self.embed(wine), "vector_embedding": self.embed(text)}
        return df

    def wip_add_to_df(self, df):
        for entry in ENTRIES:
            df = self.insert_embedding_to_def(df, entry)
        return df

    def vector_similarity(self, df, text, column):
        vec = self.embed(text)
        df["similarity"] = df[column].apply(lambda x: cosine_similarity([x], [vec]))
        df = df.sort_values(by=["similarity"], ascending=False).reset_index()
        return df.loc[0]

    def retrieve(self, text):
        most_relevant_narrative = self.vector_similarity(self.df, text, column="vector_embedding")
        most_relevant_wine = self.vector_similarity(self.df, self.extract_wine_from_query(text), column="wine_embedding")
        # print(f"You asked for        :   {text}")
        # print(f"The best narrative is:   {most_relevant_narrative['text']}")
        # print(f"The best wine is:    :   {most_relevant_wine['wine']}: {most_relevant_wine['text']}")
        # print()
        return {"most_relevant_narrative": most_relevant_narrative['text'],
                "most_relevant_wine": most_relevant_wine['text']}

    def llm_wine_explanation(self, text):
        d = self.retrieve(text)
        narrative = d["most_relevant_narrative"]
        wine = d["most_relevant_wine"]
        prompt = f"""{text}
        Some wines I've tried before information:
        {narrative}

        {wine}"""

        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        print(f"You asked: {prompt}")
        print()
        print(completion["choices"][0]["message"]["content"])
        print()
        print("===========================")
        print()


if __name__ == "__main__":
    ws = WineSnob()
    print(ws.df)
    for query in QUERIES:
        ws.llm_wine_explanation(query)
    del ws
