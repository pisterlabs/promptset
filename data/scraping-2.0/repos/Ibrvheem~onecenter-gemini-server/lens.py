import os
from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
import pinecone
import google.generativeai as genai

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

tru = Tru()

pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_API_ENV'),
    )

class RAG_from_scratch:
    @instrument
    def retrieve(self, query: str) -> str:
        """
        Retrieve relevant text from pinecone.
        """
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        docsearch = Pinecone.from_existing_index(index_name=os.getenv("DEFAULT_INDEX"), embedding=embeddings)
        retriever=docsearch.as_retriever()
        retrieved = retriever.get_relevant_documents(query)
        context = "\n".join([document.page_content for document in retrieved])
        return context

    
    @instrument
    def generate_completion(self, query:str, context: list) -> str:
        """
        Generate answer from context.
        """
        model = genai.GenerativeModel('gemini-pro')

        system_message = f"""
            "You are a helpful customer support agent."
            "You provide assistant to callers about {os.getenv("DEFAULT_INDEX").capitalize()}"
            "You can ask questions to help you understand and diagnose the problem."
            "If you are unsure of how to help, you can suggest the client to go to the nearest {os.getenv("DEFAULT_INDEX").capitalize()} office."
            "Try to sound as human as possible"
            "Make your responses as concise as possible"
            "Your response must be in plain text"
            """

        messages = [
            {
                'role':'user',
                'parts': [system_message]
                }
        ]

        messages[0]['parts'].append(f"Based on our conversation and the context below: {query}\n Context: {context}")
    
        response = model.generate_content(messages)
        return response.text
        

    @instrument
    def query(self, query:str ) -> str:
        context = self.retrieve(query)
        completion = self.generate_completion(query, context)
        return completion
    
rag = RAG_from_scratch()

from trulens_eval import Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI

import numpy as np
import streamlit as st

# Initialize provider class
fopenai = fOpenAI()

grounded = Groundedness(groundedness_provider=fopenai)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_qa_relevance = (
    Feedback(fopenai.relevance_with_cot_reasons, name = "Answer Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on_output()
)

# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(fopenai.qs_relevance_with_cot_reasons, name = "Context Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets.collect())
    .aggregate(np.mean)
)

from trulens_eval import TruCustomApp
tru_rag = TruCustomApp(rag,
    app_id = 'OnceCenter RAG',
    feedbacks = [f_groundedness, f_qa_relevance, f_context_relevance])

with tru_rag as recording:
    rag.query("Do you offer door deliveries?")
    rag.query("How fast can I get an item delivered?")

tru.get_leaderboard(app_ids=['OnceCenter RAG'])

tru.run_dashboard()