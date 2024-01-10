from datetime import datetime
from langchain.adapters.openai import convert_openai_messages
from langchain.chat_models import ChatOpenAI
import json5 as json


sample_json = """
{
  "title": title of the article,
  "date": today's date,
  "paragraphs": [
    "paragraph 1",
    "paragraph 2",
    "paragraph 3",
    "paragraph 4",
    "paragraph 5",
    ],
    "summary": "2 sentences summary of the article"
}
"""

class WriterAgent:
    def __init__(self):
        pass

    def writer(self, query: str, sources: list):
        """
        Curate relevant sources for a query
        :param query:
        :param sources:
        :return:
        """

        prompt = [{
            "role": "system",
            "content": "You are a newspaper writer. Your sole purpose is to write a well-written article about a "
                       "topic using a list of articles.\n "
        }, {
            "role": "user",
            "content": f"Today's date is {datetime.now().strftime('%d/%m/%Y')}\n."
                       f"Query or Topic: {query}"
                       f"{sources}\n"
                       f"Your task is to write a critically acclaimed article for me about the provided query or "
                       f"topic based on the sources.\n "
                       f"Please return nothing but a JSON in the following format:\n"
                       f"{sample_json}\n "

        }]

        lc_messages = convert_openai_messages(prompt)
        response = ChatOpenAI(model='gpt-4', max_retries=1).invoke(lc_messages).content
        return json.loads(response)


    def run(self, query: str, sources: list):
        return self.writer(query, sources)