import os

# create OpenAI client
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")

# ------------------- Quick build RAG part --------------------- #

# sample docs for RAG
university_info = """
The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world.
"""

import chromadb
from chromadb.utils import embedding_functions

embedding_func = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key)

embedded_data = embedding_func([university_info])

vector_store = chromadb.Client().create_collection(name="university", embedding_function=embedding_func)

vector_store.add(
    embeddings=embedded_data,
    documents=[university_info],
    metadatas=[{'source': 'university info'}],
    ids=["id1"]
)


from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument
tru = Tru()
tru.reset_database()

oai_client = OpenAI()

class RAG_from_scratch:
    @instrument
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = vector_store.query(
        query_texts=query,
        n_results=2
    )
        return results['documents'][0]

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=
            [
                {"role": "user",
                "content": 
                f"We have provided context information below. \n"
                f"---------------------\n"
                f"{context_str}"
                f"\n---------------------\n"
                f"Given this information, please answer the question: {query}"
                }
            ]
        ).choices[0].message.content
        return completion

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query)
        completion = self.generate_completion(query, context_str)
        return completion

rag = RAG_from_scratch()

# ------------------- End RAG part --------------------- #

# Define TruLens feedback functions

from trulens_eval import Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
import numpy as np

fopenai = fOpenAI() # defualt model gpt-3.5-turbo

grouded = Groundedness(groundedness_provider=fopenai)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(grouded.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
    .aggregate(grouded.grounded_statements_aggregator)
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

# Construct the app and run
from trulens_eval import TruCustomApp

tru_rag = TruCustomApp(
    rag,
    app_id = "RAG baseline",
    feedbacks = [f_groundedness, f_qa_relevance, f_context_relevance]
)

with tru_rag as recording:
    rag.query("When was the University of Washington founded?")

tru.run_dashboard()