import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Pinecone
import pinecone
import re
from pymongo import MongoClient


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')

class Vectorbase():

    def __init__(self) -> None:
        
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_API_ENV,
            )
        
        index_name = "sampledata"
        index = pinecone.Index(index_name)

        # print(pinecone.list_indexes())
        # print(index.describe_index_stats())

        embedding = OpenAIEmbeddings(
            model='text-embedding-ada-002',
            openai_api_key=OPENAI_API_KEY
        )

        text_field = "text"
        self.vectorstore = Pinecone(
            index, embedding, text_field
        )

    def get_docs(self, prompt):
        return self.vectorstore.similarity_search(prompt)
    
    def get_retriever(self):
        return self.vectorstore.as_retriever()

class Chain():
    
    def __init__(self):

        self.retriever = Vectorbase()
        
        self.condense_template = PromptTemplate(
            template="""Given the chat history and a follow up question, summarize it all into 1 standalone question.
Some examples of good standalone questions: "Do you have any black blazers with a peaked lapel?", "What silk suits go well to a forest-themed occassion?", "What blue suits do you have under $100?"

Chat History:
{chat_history}
Follow up question: {question}
Standalone Question:""",
            input_variables=["chat_history", "question"]
        )

        self.condense_llm = ChatOpenAI(
                temperature=0.0,
                model_name='gpt-3.5-turbo', verbose=True)
#         """Using the following product listings, answer the follow up question. Respond to the question like a conversational suit-store employee would and, when appropriate or necessary, ask informational questions or provide information to add to the conversation.

# Product listings:
# {documents}
# Follow Up Question: {question}
# Response:"""
        self.qa_template = PromptTemplate(
            template="""Using the following product listings, answer the question as a witty, joke-making, and descriptive men's suit store employee. You keep your responses short, and explain why you're recommendations are relevant to what the customer is asking for.
Product listings:
{documents}
Question: {question}
Answer:""",
            input_variables=["documents", "question"]
        )

        self.qa_llm = ChatOpenAI(
                temperature=0.0,
                model_name='gpt-3.5-turbo', verbose=True)

    # def parse_history(self, hist):
    #     convo = ""
    #     for r in hist:
    #         convo = convo + f"Human: {r[0]}\n"
    #         convo = convo + f"AI: {r[1]}\n"
    #     return convo

    def parse_history(self, hist):
        hist = hist.split("+$+")
        print("History:", hist)
        convo = ""
        i = 0
        while i < (len(hist) - 1):
            convo = convo + f"Human: {hist[i]}\n"
            convo = convo + f"AI: {hist[i + 1]}\n"
            i += 2
        return convo

    
    def parse_documents(self, docs):
        documents = ""
        for d in docs:
            documents = documents + f"{d.page_content}"
            documents = documents + "\n\n"
        return documents
    

    # extracts item names from data and searches atlas database for results
    # returns a list of dictionaries, each dictionary:
    # { 'name': 'Charcoal Soft Jacket', 'img-url': 'https://cdn.shoplightspeed.com/shops/639523/files/57295112/800x1067x3/charcoal-soft-jacket.jpg'}
    def search_items(self, documents, records):
        pattern = r"Name: (.*?)\nPrice"
        extracted_texts = re.findall(pattern, documents)
        print("Recommendations:",extracted_texts)

        try:
            result = []
            # for item in extracted_texts:
            query = {"name": {"$in": extracted_texts}}
            record = records.find(query)
            for r in record:
                del r ["_id"]
                result.append(r)
            return result
        except Exception as e:
            return None
        
    # input is the user query as a dictionary
    # records is a pymongo collection object (query-able object)
    def get_response(self, input, records):
        # unload input
        print('-------------')
        chat_history = input["chat_history"]
        query = input["question"]
        print("Query:", query)
        # parse history list into usable string
        hist = ""
        if (chat_history != None):
            hist = self.parse_history(chat_history)

        # generate standalone question
        condense_formatted = self.condense_template.format(question=query, chat_history=hist)
        standalone = self.condense_llm.predict(condense_formatted)
        print("Standalone:", standalone)
        # retrieve documents
        documents = self.retriever.get_docs(standalone)
        documents = self.parse_documents(documents)

        # get item info from mongodb database
        sidebar_data = self.search_items(documents, records)
        # general final answer
        qa_formatted = self.qa_template.format(question=standalone, documents=documents)
        answer = self.qa_llm.predict(qa_formatted)
        print("Answer:", answer)
        return answer, hist, sidebar_data
        
def main():
    uri = "mongodb+srv://u1:u1@cluster0.4cpubm9.mongodb.net/?retryWrites=true&w=majority"
    # Create a new client and connect to the server
    client = MongoClient(uri)
    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

    db = client.get_database('LOR')
    records = db.get_collection('item-data')

    c = Chain()
    # history = [("I am looking for a suit", "What color suit are you looking for?")]
    # query = "Do you have a black one?"

    history = "I am going to a party that is forest themed and am planning an outfit+$+Thats great what do you need help with?"
    query = "What suits would fit the theme?"

    input = {
        "question": query,
        "chat_history": history
    }

    r = c.get_response(input, records)


    # a = Chain()
    # # a.response("I am looking for a blazer.")
    # print(a.response("What suits do you have in black?", []))

    # # print(a.response("What is the phone number of the store?"))

    # # V = Vectorbase()
    # # print(V.search())


if __name__ == "__main__":
    main()
