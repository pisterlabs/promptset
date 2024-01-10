import os
import re
import json
from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import tiktoken
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")

class insight_Generator():

    def __init__(self):
        self.llm = OpenAI()
        self.text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
        self.prompt_template_insight = """You are a top analyst at an investment bank whose daily job is to analyse news articles to draw insights about the movement of stock prices. Generate investment related insights that you would use in your daily job from the article below. 
        THEN, generate a list of the top 20 tags that could affect the stock price.
        THEN, generate a list of relevant invest risks relevant to the article ONLY from this list of risks: "Interest Rate Risk, Market Risk, Credit Risk, Liquidity Risk, Operational Risk, Reinvestment Risk, Currency Risk, Inflation Risk, Political Risk, Business Risk, Volatility Risk, Concentration Risk, Longevity Risk, Model Risk, Counterparty Risk". return NIL if no risk is relevant.
        

        {text}


        --EXAMPLE--
        INSIGHTS: "Insight here"
        TAGS: "tag 1, tag 2, tag 3, tag 4, tag 5, tag 6, tag 7, tag 8, tag 9, tag 10, tag 11, tag 12, tag 13, tag 14, tag 15, tag 16, tag 17, tag 18, tag 19, tag 20"
        RISKS: "risk 1, risk 2, risk 3..."
        """
        self.prompt_template_sum = """Generate a detailed summary of all the insights below:
        

        {text}


        --EXAMPLE--
        INSIGHT SUMMARY: "Insight summary here"
        """


    def data_splitter(self, data):
        texts = self.text_splitter.split_text(data)
        return texts



    def token_counter(self, string: str, encoding_name: str) -> int:
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens



    def run_chain(self, news_articles):
        insight_list = []
        tags_list = []

        for i, article in enumerate(news_articles):
            texts = article['content']
            tkn_count = self.token_counter(texts, "gpt-3.5-turbo")
            if tkn_count > 4095:
                texts = self.data_splitter(texts)
            else:
                texts = [texts]
            docs = [Document(page_content=t) for t in texts]
            PROMPT = PromptTemplate(template=self.prompt_template_insight, input_variables=["text"])
            chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=PROMPT)
            chain({"input_documents": docs}, return_only_outputs=False)
            result = chain.run(input_documents=docs)

            start = result.find("INSIGHT:") + len("INSIGHT:")
            end = result.find("TAGS:")
            start2 = end + len("TAGS:")
            end2 = result.find("RISKS:")
            start3 = end2 + len("RISKS:")
 
            insight = result[start:end].strip()
            tags_raw = result[start2:end2].strip().rstrip(".")
            risks_raw = result[start3:].strip().rstrip(".")
            news_articles[i]['tags'] = tags_raw
            news_articles[i]['risks'] = risks_raw

            insight_list.append(insight)

        sum_insights_raw = "\n\n".join(insight_list)
        tkn_count = self.token_counter(sum_insights_raw, "gpt-3.5-turbo")
        if tkn_count > 4095:
            texts = self.data_splitter(sum_insights_raw)
        else:
            texts = [sum_insights_raw]
        docs = [Document(page_content=t) for t in texts]
        PROMPT_SUM = PromptTemplate(template=self.prompt_template_sum, input_variables=["text"])
        chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=PROMPT_SUM)
        sum_insights = chain.run(input_documents=docs)
        return sum_insights, news_articles


def insight(news_articles):
    insights, news_articles = insight_Generator().run_chain(news_articles)
    return insights, news_articles