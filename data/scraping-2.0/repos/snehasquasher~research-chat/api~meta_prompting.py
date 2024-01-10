# -*- coding: utf-8 -*-
import openai
import nest_asyncio

import os

from pathlib import Path
from llama_hub.file.pdf.base import PDFReader
from llama_hub.file.unstructured.base import UnstructuredReader
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index import Document
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import IndexNode
from llama_index.evaluation import DatasetGenerator, QueryResponseDataset
from llama_index.node_parser import SimpleNodeParser
import random

from llama_index.evaluation.eval_utils import get_responses
from llama_index.evaluation import CorrectnessEvaluator, BatchEvalRunner
QA_PROMPT_KEY = "response_synthesizer:text_qa_template"

from llama_index.llms import OpenAI
from llama_index.prompts import PromptTemplate
from copy import deepcopy

import numpy as np
import pickle
import asyncio


def createMetaTemplate():
    qa_tmpl_str = (
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_tmpl = PromptTemplate(qa_tmpl_str)

    #print(query_engine.get_prompts()[QA_PROMPT_KEY].get_template())

    meta_tmpl_str = """\
    Your task is to generate the instruction <INS>. Below are some previous instructions with their scores.
    The score ranges from 1 to 5.

    {prev_instruction_score_pairs}

    Below we show the task. The <INS> tag is prepended to the below prompt template, e.g. as follows:

    ```
    <INS>
    {prompt_tmpl_str}
    ```

    The prompt template contains template variables. Given an input set of template variables, the formatted prompt is then given to an LLM to get an output.

    Some examples of template variable inputs and expected outputs are given below to illustrate the task. **NOTE**: These do NOT represent the \
    entire evaluation dataset.

    {qa_pairs_str}

    We run every input in an evaluation dataset through an LLM. If the LLM-generated output doesn't match the expected output, we mark it as wrong (score 0).
    A correct answer has a score of 1. The final "score" for an instruction is the average of scores across an evaluation dataset.
    Write your new instruction (<INS>) that is different from the old ones and has a score as high as possible.

    Instruction (<INS>): \
    """

    meta_tmpl = PromptTemplate(meta_tmpl_str)
    return meta_tmpl


async def metaPrompt():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    base_nodes = loadPDFs()
    rag_service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", api_key = os.getenv("OPENAI_API_KEY"))
    )
    index = VectorStoreIndex(base_nodes, service_context=rag_service_context)
    query_engine = index.as_query_engine(similarity_top_k=3)
    eval_dataset = await createDataset(base_nodes)
    full_qr_pairs = eval_dataset.qr_pairs
    exemplar_qr_pairs, eval_qr_pairs, evaluator_dict = createEvaluateInfo(full_qr_pairs)
    batch_runner = BatchEvalRunner(evaluator_dict, workers=2, show_progress=True)

    llm = OpenAI(model="gpt-3.5-turbo", api_key = os.getenv("OPENAI_API_KEY"))
    meta_tmpl = createMetaTemplate()
    # define and pre-seed query engine with the prompt
    query_engine = index.as_query_engine(similarity_top_k=2)
    # query_engine.update_prompts({QA_PROMPT_KEY: qa_tmpl})

    # get the base qa prompt (without any instruction prefix)
    base_qa_prompt = query_engine.get_prompts()[QA_PROMPT_KEY]

    initial_instr = """\
    You are a QA assistant.
    Context information is below. Given the context information and not prior knowledge, \
    answer the query. \
    """

    # this is the "initial" prompt template
    # implicitly used in the first stage of the loop during prompt optimization
    # here we explicitly capture it so we can use it for evaluation
    old_qa_prompt = get_full_prompt_template(initial_instr, base_qa_prompt)

    meta_llm = OpenAI(model="gpt-3.5-turbo", api_key = os.getenv("OPENAI_API_KEY"))

    new_instr, prev_instr_score_pairs = await optimize_prompts(
        query_engine,
        initial_instr,
        base_qa_prompt,
        meta_tmpl,
        meta_llm,  # note: treat llm as meta_llm
        batch_runner,
        eval_qr_pairs,
        exemplar_qr_pairs,
        num_iterations=5,
    )


    new_qa_prompt = query_engine.get_prompts()[QA_PROMPT_KEY]
    

    #pickle.dump(prev_instr_score_pairs, open("prev_instr_score_pairs.pkl", "wb"))


    full_eval_qs = [q for q, _ in full_qr_pairs]
    full_eval_answers = [a for _, a in full_qr_pairs]

    print(full_eval_qs, full_eval_answers)

    query_engine.update_prompts({QA_PROMPT_KEY: old_qa_prompt})
    avg_correctness_old = await get_correctness(
        query_engine, full_qr_pairs, batch_runner
    )

    print(old_qa_prompt, avg_correctness_old)

    query_engine.update_prompts({QA_PROMPT_KEY: new_qa_prompt})
    avg_correctness_new = await get_correctness(
        query_engine, full_qr_pairs, batch_runner
    )

    print(new_qa_prompt, avg_correctness_new)
    return new_qa_prompt.template

def loadPDFs():
    print("Begin loading PDFs")
    loader = PDFReader()
    docs0 = []
    upload_directory = "./user-uploads"
    print(os.listdir(upload_directory))

    uploaded_files = [f for f in os.listdir(upload_directory) if os.path.isfile(os.path.join(upload_directory, f))]
    for upload_file in uploaded_files:
        print("reading ", upload_file)
        docs0.extend(loader.load_data(file=os.path.join(upload_directory, upload_file)))

    doc_text = "\n\n".join([d.get_content() for d in docs0])
    docs = [Document(text=doc_text)]


    node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
    base_nodes = node_parser.get_nodes_from_documents(docs)
    return base_nodes


async def createDataset(base_nodes):

    eval_service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4", api_key = os.getenv("OPENAI_API_KEY")))

    dataset_generator = DatasetGenerator(
        base_nodes[:20],
        service_context=eval_service_context,
        show_progress=True,
        num_questions_per_chunk=3,
    )

    eval_dataset = await dataset_generator.agenerate_dataset_from_nodes(num=60)

    eval_dataset.save_json("../metaprompt_eval_dataset.json")

    # optional
    eval_dataset = QueryResponseDataset.from_json(
        "../metaprompt_eval_dataset.json"
    )

    return eval_dataset

def createEvaluateInfo(full_qr_pairs):
    num_exemplars = 2
    num_eval = 40
    exemplar_qr_pairs = random.sample(full_qr_pairs, num_exemplars)
    eval_qr_pairs = random.sample(full_qr_pairs, num_eval)

    eval_service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-3.5-turbo", api_key = os.getenv("OPENAI_API_KEY"))
    )

    evaluator_c = CorrectnessEvaluator(service_context=eval_service_context)
    evaluator_dict = {
        "correctness": evaluator_c,
    }

    return exemplar_qr_pairs, eval_qr_pairs, evaluator_dict
    

async def get_correctness(query_engine, eval_qa_pairs, batch_runner):
    # then evaluate
    # TODO: evaluate a sample of generated results
    eval_qs = [q for q, _ in eval_qa_pairs]
    eval_answers = [a for _, a in eval_qa_pairs]
    pred_responses = get_responses(eval_qs, query_engine, show_progress=True)

    eval_results = await batch_runner.aevaluate_responses(
        eval_qs, responses=pred_responses, reference=eval_answers
    )
    avg_correctness = np.array(
        [r.score for r in eval_results["correctness"]]
    ).mean()
    return avg_correctness


def format_meta_tmpl(
    prev_instr_score_pairs,
    prompt_tmpl_str,
    qa_pairs,
    meta_tmpl,
):
    """Call meta-prompt to generate new instruction."""
    # format prev instruction score pairs.
    pair_str_list = [
        f"Instruction (<INS>):\n{instr}\nScore:\n{score}"
        for instr, score in prev_instr_score_pairs
    ]
    full_instr_pair_str = "\n\n".join(pair_str_list)

    # now show QA pairs with ground-truth answers
    qa_str_list = [
        f"query_str:\n{query_str}\nAnswer:\n{answer}"
        for query_str, answer in qa_pairs
    ]
    full_qa_pair_str = "\n\n".join(qa_str_list)

    fmt_meta_tmpl = meta_tmpl.format(
        prev_instruction_score_pairs=full_instr_pair_str,
        prompt_tmpl_str=prompt_tmpl_str,
        qa_pairs_str=full_qa_pair_str,
    )
    return fmt_meta_tmpl

def get_full_prompt_template(cur_instr: str, prompt_tmpl):
    tmpl_str = prompt_tmpl.get_template()
    new_tmpl_str = cur_instr + "\n" + tmpl_str
    new_tmpl = PromptTemplate(new_tmpl_str)
    return new_tmpl


def _parse_meta_response(meta_response: str):
    return str(meta_response).split("\n")[0]


async def optimize_prompts(
    query_engine,
    initial_instr: str,
    base_prompt_tmpl,
    meta_tmpl,
    meta_llm,
    batch_runner,
    eval_qa_pairs,
    exemplar_qa_pairs,
    num_iterations: int = 5,
):
    prev_instr_score_pairs = []
    base_prompt_tmpl_str = base_prompt_tmpl.get_template()

    cur_instr = initial_instr
    for idx in range(num_iterations):
        # TODO: change from -1 to 0
        if idx > 0:
            # first generate
            fmt_meta_tmpl = format_meta_tmpl(
                prev_instr_score_pairs,
                base_prompt_tmpl_str,
                exemplar_qa_pairs,
                meta_tmpl,
            )
            meta_response = meta_llm.complete(fmt_meta_tmpl)
            print(fmt_meta_tmpl)
            print(str(meta_response))
            # Parse meta response
            cur_instr = _parse_meta_response(meta_response)

        # append instruction to template
        new_prompt_tmpl = get_full_prompt_template(cur_instr, base_prompt_tmpl)
        query_engine.update_prompts({QA_PROMPT_KEY: new_prompt_tmpl})

        avg_correctness = await get_correctness(
            query_engine, eval_qa_pairs, batch_runner
        )
        prev_instr_score_pairs.append((cur_instr, avg_correctness))

    # find the instruction with the highest score
    max_instr_score_pair = max(
        prev_instr_score_pairs, key=lambda item: item[1]
    )

    # return the instruction
    return max_instr_score_pair[0], prev_instr_score_pairs

async def runMetaPrompt():
    #return 'Given the context information and not prior knowledge, provide a detailed and comprehensive response to the query.\nContext information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query_str}\nAnswer: '
    nest_asyncio.apply()
    try:
        newPrompt = await metaPrompt()
        #newPrompt = asyncio.run(metaPrompt())
        print(newPrompt)
        return newPrompt
    except Exception as e:
        print(e)
        return ""

