from dotenv import load_dotenv

load_dotenv()

import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI

tru = Tru()

from llama_index import VectorStoreIndex
from llama_index.readers import SimpleDirectoryReader

documents = SimpleDirectoryReader(input_dir=os.getcwd(), input_files=['data.json']).load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

import numpy as np

# Initialize provider class
openai = OpenAI()

grounded = Groundedness(groundedness_provider=OpenAI())

# Define a groundedness feedback function
f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
    TruLlama.select_source_nodes().node.text.collect()
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai.qs_relevance).on_input().on(
    TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

tru_query_engine_recorder = TruLlama(query_engine,
    app_id='LlamaIndex_App9',
    feedbacks=[f_qa_relevance, f_qs_relevance])

# or as context manager
with tru_query_engine_recorder as recording:
    response = query_engine.query("I'm working on a tech startup. I'm looking to meet people all over the world who have work-related achievements, and also current challenges. I don't care about their location, and industry.")
    print(response)

# open a local streamlit app to explore
tru.run_dashboard()

# LlamaIndex_App1 - "Whom it would make sense for me to network with? My Location: Barcelona. My Company description: I'm working on a data analytics SaaS company."
# LlamaIndex_App2 - "Whom it would make sense for me to network with, and meet via a video call? My Location: Barcelona. My Company description: I'm working on a data analytics SaaS company."
# LlamaIndex_App3 - "My Location: Barcelona. My Company description: I'm working on a data analytics SaaS company. I'm looking to network (using videocalls) with smart people who work on innovative solutions."
# LlamaIndex_App4 - "My Location: Barcelona. My Company description: I'm working on a data analytics SaaS company. I'm looking to meet over a virtual coffee people with growth mindset."
# LlamaIndex_App5 - "My Location: Barcelona. My Company description: I'm working on a data analytics SaaS company. I'm looking to meet over a virtual coffee with people who have past achievements at their companies, but also have unsolved current challenges."
# LlamaIndex_App6 - "My Location: Barcelona. My Company description: I'm working on a tech startup. I'm looking to meet over a virtual coffee with people who have past achievements at their companies, but also have unsolved current challenges."
# LlamaIndex_App7 - "I live in Barcelona, and I'm working on a tech startup. I'm looking to meet over a virtual coffee with people all over the world who have past achievements at their companies, but also have unsolved current challenges."
# LlamaIndex_App8 - "I live in Barcelona, and I'm working on a tech startup. I'm looking to meet people all over the world who had achievements at their companies in the past, but also have unsolved current challenges. I don't care about their location, and industry. I'm looking to meet over a virtual coffee (using videocalls)."
# LlamaIndex_App9 - "I'm working on a tech startup. I'm looking to meet people all over the world who have work-related achievements, and also current challenges. I don't care about their location, and industry."