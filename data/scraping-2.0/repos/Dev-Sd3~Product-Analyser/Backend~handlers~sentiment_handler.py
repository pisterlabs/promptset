from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
import pandas as pd
import os.path

'''
> Sentiment Analysis cost analysis:
The average best buy review contains 28 words,
The response is always 1 or 2 words per prompt.
Per OpenAI's website:
- 750 words = 1000 tokens, 1 word ~ 1.25 tokens
- GPT-3.5 Turbo charges 0.0015$ per 1000 tokens, thus,
- 750 words cost 0.0015$, thus 1-word costs 0.000002$


> Sentiment analysis handler:

Sentiment analysis is done using OpenAI's LLM with template-based prompts
after manually testing different prompt templates in chatGPT, the best results were using this lenthy template:

"You are given a review which you will have to analyze using only one of the following quantifiers: Very positive, positive, neutral, negative, very negative. 
if the query given is not an opinion or its a question, reply with none, here is the review: {query}, remember only reply with one of the quantifiers given with nothing else

Thus we get one of the above sentiments or none if the review is a question (and there is alot of those)


> Cost:

Currently the code only analyses bestbuy reviews since they are typically shorter, fewer and more concise, the average cost of analysing one best buy review is around: 0.00012

While reddit data is much larger and lenthier, the cost of analysing 1 reddit comment is around:  0.000152

Thus, only best buy sentiment is analysed, reddit data analysis can be implemented by switching the analysed table name and review to comment

'''


class SentimentAnalysis:
    def __init__(self, url):
        self.key = self.getKey()
        self.llm = OpenAI(openai_api_key=self.key, temperature=0.2)
        self.connection_url = url
        self.engine = create_engine(self.connection_url)

    def getKey(self):
        try:
            assert os.path.isfile('Backend/Credentials/key.key')
            assert os.path.isfile('Backend/Credentials/encrypted.key')

            with open("Backend/Credentials/key.key", "rb") as key_file:
                key = key_file.read()
            with open("Backend/Credentials/encrypted.key", "rb") as encrypted_message:
                encrypted_message = encrypted_message.read()

            fernet = Fernet(key)
            decrypted_message = fernet.decrypt(encrypted_message)
            OPENAI_API_KEY = decrypted_message.decode()
            return OPENAI_API_KEY

        except:
            print("couldnt open key")

    def getSentiment(self, review):
        prompt = PromptTemplate.from_template(
            "You are given a review which you will have to analyze using only one of the following quantifiers: Very positive, positive, neutral, negative, very negative. if the query given is not an opinion or its a question, reply with none, here is the review: {query}, remember only reply with one of the quantifiers given with nothing else")
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(review).replace('\n\n', '').lower()
        return response

    def getSentimentEntry(self, query):
        print("here")
        with self.engine.connect() as db:
            df = pd.read_sql(f"""SELECT * FROM "ProductAnalysis"."stgRedditPosts"
                                WHERE "ProductAnalysis"."stgRedditPosts"."query" = '{query}'
                            """, db)
        df['Sentiment'] = df['comment'].map(lambda x: self.getSentiment(x))

        vp, p, neut, n, vn, none = 0, 0, 0, 0, 0, 0
        for sentiment in df['Sentiment'].tolist():
            if sentiment == 'very positive':
                vp += 1
            elif sentiment == 'positive':
                p += 1
            elif sentiment == 'neutral':
                neut += 1
            elif sentiment == 'negative':
                n += 1
            elif sentiment == 'very negative':
                vn += 1
            else:
                none += 1
        product = df['query'].tolist()[0]
        origin = 'Reddit'
        total = vp+p+neut+n+vn+none

        return [query, origin,total, vp, p, neut, n, vn, none]
    
    def getQueries(self):
        with self.engine.connect() as db:
            df = pd.read_sql(
                'SELECT DISTINCT "ProductAnalysis"."stgRedditPosts"."query" from "ProductAnalysis"."stgRedditPosts"', db)
        return df

    def getDataframe(self):
        keys = self.getQueries()
        all_entries = []
        for key in keys['query'].tolist():
            all_entries.append(self.getSentimentEntry(key))
        df = pd.DataFrame(all_entries, columns=['query', 'origin','totalReviews', 'Very positive', 
                                                'positive', 'neutral', 'negative', 'very negative', 'none'])
        return df

    def writeToPostGre(self):
        df = self.getDataframe()
        with self.engine.connect() as db:
            df.to_sql('aggSentimentAnalysis', db, if_exists='replace',
                      index=False, schema='ProductAnalysis')
        return df