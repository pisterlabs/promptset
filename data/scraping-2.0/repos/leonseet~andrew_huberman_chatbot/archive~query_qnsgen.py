from llama_index.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    LLMPredictor,
    Response,
)
from llama_index.llms import OpenAI
from llama_index import Document
from llama_index.prompts import PromptTemplate
from llama_index.postprocessor import KeywordNodePostprocessor
from llama_index.schema import Node, NodeWithScore
from llama_index.extractors import QuestionsAnsweredExtractor

import pandas as pd
from dotenv import load_dotenv
from libs.index import initialize_chroma_collection
from libs.query import get_llm

load_dotenv()

chroma_collection = initialize_chroma_collection()
docs = chroma_collection.get(offset=10, limit=10)
docs = [Document(text=t) for t in docs["documents"]]

# print(docs.keys())
# print(docs["documents"])
# print(len(docs["documents"]))

llm = get_llm("gpt-3.5-turbo")
# llm = get_llm("meta-llama/Llama-2-70b-chat-hf")

service_context = ServiceContext.from_defaults(
    llm=llm,
)


template = (
    "Context information is below."
    "\n---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and no prior knowledge, generate only scientific questions based on the below query. Generate <None> if no scientific question can be generated.\n"
    "{query_str}\n"
)

text_question_template = PromptTemplate(template)

data_generator = DatasetGenerator.from_documents(
    documents=docs,
    service_context=service_context,
    num_questions_per_chunk=2,
    text_question_template=text_question_template,
    show_progress=True,
    # exclude_keywords=["Inside Tracker", "premium channel", "podcast"],
    # required_keywords=["?"],
)


# print(docs)
# print(data_generator.get_prompts())

eval_questions = data_generator.generate_questions_from_nodes()


postprocessor = KeywordNodePostprocessor(
    required_keywords=["?"],
    exclude_keywords=["Inside Tracker", "premium channel", "podcast", "live events"],
)
nodes = [NodeWithScore(node=Node(text=t), score=0) for t in eval_questions]
eval_questions = postprocessor.postprocess_nodes(nodes)
eval_questions = [node.text for node in eval_questions]

print(eval_questions)
print(len(eval_questions))
