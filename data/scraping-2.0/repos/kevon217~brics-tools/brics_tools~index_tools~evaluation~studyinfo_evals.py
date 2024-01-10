import os
from dotenv import load_dotenv
import openai
from pathlib import Path

from llama_index import (
    Document,
    download_loader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import OpenAI

from brics_tools.utils import helper
from brics_tools.utils.llm_utils.tokens import openai_api_calculate_cost
from brics_tools.index_tools.loaders.studyinfo_loader import StudyInfoLoader
from brics_tools.index_tools.document_creators.studyinfo_document_creator import (
    StudyInfoDocumentCreator,
)
from brics_tools.index_tools.node_parsers.studyinfo_node_parser import (
    StudyInfoNodeParser,
)
from brics_tools.index_tools.index_managers.studyinfo_vectorstore_manager import (
    StudyInfoVectorStoreIndexManager,
)
from brics_tools.index_tools.index_managers.studyinfo_summary_manager import (
    StudyInfoSummaryIndexManager,
)
from brics_tools.index_tools.index_loaders.studyinfo_index_loader import (
    StudyInfoVectorStoreIndexLoader,
)
from brics_tools.index_tools.query_engines.studyinfo_query_engine import (
    StudyInfoQueryEngine,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

cfg = helper.compose_config(
    config_path="../configs/",
    config_name="config_studyinfo",
    overrides=[],
)


# create a global service context
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0)
)
# set_global_service_context(service_context)

# GENERATE DATASET

studyinfo_loader = StudyInfoLoader(cfg.loaders.studyinfo_loader)
studyinfo_loader.load_studies()
df_studyinfo = studyinfo_loader.df_studyinfo
studyinfo_doc_creator = StudyInfoDocumentCreator(
    cfg.document_creators.studyinfo_document
)
studyinfo_docs = studyinfo_doc_creator.create_documents(studyinfo_loader.df_studyinfo)
print(studyinfo_docs[0])

import tiktoken


def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


all_text = ""

for doc in studyinfo_docs:
    all_text += doc.text

giant_document = Document(text=all_text)

usage = {}
# usage['prompt_tokens'] = num_tokens_from_string(giant_document.text, "gpt-4")
usage["prompt_tokens"] = num_tokens_from_string(giant_document.text, "gpt-3.5-turbo")
usage["completion_tokens"] = 2000
usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

total_cost = openai_api_calculate_cost(usage, model="gpt-3.5-turbo-16k")
total_cost = openai_api_calculate_cost(usage, model="gpt-4-8k")

from llama_index.evaluation import DatasetGenerator

service_context_generator = ServiceContext.from_defaults(
    llm=OpenAI(
        model=cfg.evaluators.studyinfo_question_dataset.service_context.generator.llm.llm_kwargs.model_name,
        temperature=cfg.evaluators.studyinfo_question_dataset.service_context.evaluator.llm.llm_kwargs.temperature,
    )
)


test = studyinfo_docs[0:3]

prompt = cfg.evaluators.studyinfo_question_dataset.prompt

data_generator = DatasetGenerator.from_documents(
    test,
    question_gen_query=prompt,
    service_context=service_context_generator,
)
question_dataset = data_generator.generate_questions_from_nodes()


# save the questions!
fp = Path(
    cfg.evaluators.studyinfo_question_dataset.storage_context.storage_path_root,
    cfg.evaluators.studyinfo_question_dataset.storage_context.filename,
)
with open(fp, "w") as f:
    for question in question_dataset:
        f.write(f"{question.strip()}\n")

import time
import asyncio
import nest_asyncio

nest_asyncio.apply()

from llama_index import Response

service_context_evaluator = ServiceContext.from_defaults(
    llm=OpenAI(
        model=cfg.evaluators.studyinfo_question_dataset.service_context.evaluator.llm.llm_kwargs.model_name,
        temperature=cfg.evaluators.studyinfo_question_dataset.service_context.evaluator.llm.llm_kwargs.temperature,
    )
)
import asyncio


def evaluate_query_engine(evaluator, query_engine, questions):
    async def run_query(query_engine, q):
        try:
            return await query_engine.aquery(q)
        except:
            return Response(response="Error, query failed.")

    total_correct = 0
    all_results = []
    for batch_size in range(0, len(questions), 5):
        batch_qs = questions[batch_size : batch_size + 5]

        tasks = [run_query(query_engine, q) for q in batch_qs]
        responses = asyncio.run(asyncio.gather(*tasks))
        print(f"finished batch {(batch_size // 5) + 1} out of {len(questions) // 5}")

        for response in responses:
            eval_result = 1 if "YES" in evaluator.evaluate(response) else 0
            total_correct += eval_result
            all_results.append(eval_result)

        # helps avoid rate limits
        time.sleep(1)

    return total_correct, all_results


from llama_index.evaluation import ResponseEvaluator
from llama_index.evaluation import QueryResponseEvaluator
import pandas as pd

# gpt-4 evaluator!
evaluator = ResponseEvaluator(service_context=service_context_evaluator)
evaluator = QueryResponseEvaluator(service_context=service_context_evaluator)

vec_eng = engine.query_engines["vector_query_engine"]
query_str = "What is TRACK-TBI studying?"
v1 = vec_eng.query(query_str)

total_correct, all_results = evaluate_query_engine(evaluator, vec_eng, question_dataset)
evaluation = evaluator.evaluate_response(query=query_str, response=v1)
print(evaluation.feedback)

print(
    f"Hallucination? Scored {total_correct} out of {len(question_dataset)} questions correctly."
)


# define jupyter display function
def eval_df(query: str, response: Response, eval_result: str) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Response": str(response),
            "Source": response.source_nodes[0].node.get_content()[:1000] + "...",
            "Evaluation Result": eval_result,
        },
        index=[0],
    )
    return eval_df


eval_df = eval_df(query_str, v1, evaluation.feedback)

import os
import random

random.seed(42)

from llama_index import ServiceContext
from llama_index.prompts import Prompt
from llama_index.llms import OpenAI
from llama_index.evaluation import DatasetGenerator

gpt4_service_context = ServiceContext.from_defaults(
    llm=OpenAI(llm="gpt-4", temperature=0)
)

question_dataset = []
if os.path.exists("question_dataset.txt"):
    with open("question_dataset.txt", "r") as f:
        for line in f:
            question_dataset.append(line.strip())
else:
    # generate questions
    data_generator = DatasetGenerator.from_documents(
        [giant_document],
        text_question_template=Prompt(
            "A sample from the LlamaIndex documentation is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Using the documentation sample, carefully follow the instructions below:\n"
            "{query_str}"
        ),
        question_gen_query=(
            "You are an evaluator for a search pipeline. Your task is to write a single question "
            "using the provided documentation sample above to test the search pipeline. The question should "
            "reference specific names, functions, and terms. Restrict the question to the "
            "context information provided.\n"
            "Question: "
        ),
        # set this to be low, so we can generate more questions
        service_context=gpt4_service_context,
    )
    generated_questions = data_generator.generate_questions_from_nodes()

    # randomly pick 40 questions from each dataset
    generated_questions = random.sample(generated_questions, 40)
    question_dataset.extend(generated_questions)

    print(f"Generated {len(question_dataset)} questions.")

    # save the questions!
    with open("question_dataset.txt", "w") as f:
        for question in question_dataset:
            f.write(f"{question.strip()}\n")
