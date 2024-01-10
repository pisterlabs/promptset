from json import JSONDecodeError
import itertools
import asyncio
import glob
import os
from datetime import datetime
from tqdm.asyncio import tqdm as tqdm_asyncio

from langchain.chains import QAGenerationChain
from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings

from ragflow.commons.prompts import QA_GENERATION_PROMPT_SELECTOR
from ragflow.utils import aload_and_chunk_docs, write_json, read_json
from ragflow.commons.configurations import QAConfigurations
from ragflow.commons.chroma import ChromaClient

import uuid
import logging

logger = logging.getLogger(__name__)


async def get_qa_from_chunk(
    chunk: Document,
    qa_generator_chain: QAGenerationChain,
) -> list[dict]:
    """Generate QA from provided text document chunk."""
    try:
        # return list of qa pairs
        qa_pairs = qa_generator_chain.run(chunk.page_content)

        # attach chunk metadata to qa_pair
        for qa_pair in qa_pairs:
            qa_pair["metadata"] = dict(**chunk.metadata)
            qa_pair["metadata"].update(
                {"id": str(uuid.uuid4()), "context": chunk.page_content}
            )

        return qa_pairs
    except JSONDecodeError:
        return []


async def agenerate_label_dataset_from_doc(
    hp: QAConfigurations,
    doc_path: str,
) -> list[dict[str, str]]:
    """Generate a pairs of QAs that are used as ground truth in downstream tasks, i.e. RAG evaluations

    Args:
        llm (BaseLanguageModel): the language model used in the QAGenerationChain
        chunks (List[Document]): the document chunks used for QA generation

    Returns:
        List[Dict[str, str]]: returns a list of dictionary of question - answer pairs
    """

    logger.debug(f"Starting QA generation process for {doc_path}.")

    # load data and chunk doc
    chunks = await aload_and_chunk_docs(hp, [doc_path])

    qa_generator_chain = QAGenerationChain.from_llm(
        hp.qa_generator_llm,
        prompt=QA_GENERATION_PROMPT_SELECTOR.get_prompt(hp.qa_generator_llm),
    )

    tasks = [get_qa_from_chunk(chunk, qa_generator_chain) for chunk in chunks]

    qa_pairs = await asyncio.gather(*tasks)
    qa_pairs = list(itertools.chain.from_iterable(qa_pairs))

    return qa_pairs


async def agenerate_label_dataset_from_docs(
    hp: QAConfigurations,
    docs_path: list[str],
) -> list[dict]:
    """Asynchronous wrapper around the agenerate_label_dataset function.

    Args:
        qa_gen_configs (dict): _description_
        docs_path (list[str]): _description_

    Returns:
        list[dict]: _description_
    """
    tasks = [agenerate_label_dataset_from_doc(hp, doc_path) for doc_path in docs_path]

    results = await tqdm_asyncio.gather(*tasks)

    qa_pairs = list(itertools.chain.from_iterable(results))

    return qa_pairs


async def aupsert_embeddings_for_model(
    qa_pairs: list[dict],
    embedding_model: Embeddings,
    user_id: str,
) -> None:
    """Embeds and upserts each generated answer into vectorstore. This is helpful if you want to run different hyperparameter runs with the same embedding model because you only have to embed these answers once. The embeddings are used during evaluation to check similarity of generated and predicted answers."""
    with ChromaClient() as CHROMA_CLIENT:
        collection_name = f"userid_{user_id}_qaid_0_{QAConfigurations.get_embedding_model_name(embedding_model)}"

        # check if collection already exists, if not create a new one with the embeddings
        if [
            collection
            for collection in CHROMA_CLIENT.list_collections()
            if collection.name.startswith(f"userid_{user_id}_")
        ]:
            logger.info(f"Collection {collection_id} already exists, skipping it.")
            return None

        collection = CHROMA_CLIENT.create_collection(
            name=collection_name,
            metadata={
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
            },
        )

        ids = [qa_pair["metadata"]["id"] for qa_pair in qa_pairs]

        # maybe async function is not implemented for embedding model
        try:
            embeddings = await embedding_model.aembed_documents(
                [qa_pair["answer"] for qa_pair in qa_pairs]
            )
        except NotImplementedError as ex:
            logger.error(
                f"Exception during eval set generation and upserting to vectorstore, {ex}"
            )
            embeddings = embedding_model.embed_documents(
                [qa_pair["answer"] for qa_pair in qa_pairs]
            )

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=[
                {
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    **qa_pair["metadata"],
                }
                for qa_pair in qa_pairs
            ],
        )

    logger.info(
        f"Upserted {QAConfigurations.get_embedding_model_name(embedding_model)} embeddings to vectorstore."
    )


async def agenerate_and_save_dataset(
    hp: QAConfigurations,
    docs_path: str,
    label_dataset_path: str,
    user_id: str,
):
    """Generate a new evaluation dataset and save it to a JSON file."""

    logger.info("Starting QA generation suite.")

    # generate label dataset
    label_dataset = await agenerate_label_dataset_from_docs(hp, docs_path)

    # During test execution: if label_dataset is empty because of test dummy LLM, we inject a real dataset for test
    if (
        os.environ.get("EXECUTION_CONTEXT") == "TEST"
        and hp.persist_to_vs
        and not label_dataset
    ):
        label_dataset = read_json(os.environ.get("INPUT_LABEL_DATASET"))

    # write eval dataset to json
    write_json(label_dataset, label_dataset_path)

    # cache answers of qa pairs in vectorstore for each embedding model provided
    if hp.persist_to_vs:
        tasks = [
            aupsert_embeddings_for_model(label_dataset, embedding_model, user_id)
            for embedding_model in hp.embedding_model_list
        ]

        await asyncio.gather(*tasks)


async def agenerate_evaluation_set(
    label_dataset_gen_params_path: str,
    label_dataset_path: str,
    document_store_path: str,
    user_id: str,
    api_keys: dict[str, str],
):
    """Entry function to generate the evaluation dataset.

    Args:
        label_dataset_gen_params (dict): _description_
        label_dataset_path (str): _description_

    Returns:
        _type_: _description_
    """
    logger.info("Checking for evaluation dataset configs.")

    label_dataset_gen_params = read_json(label_dataset_gen_params_path)

    # TODO: Only one single QA generation supported per user
    if isinstance(label_dataset_gen_params, list):
        label_dataset_gen_params = label_dataset_gen_params[-1]

    # set up QAConfiguration object at the beginning to evaluate inputs
    label_dataset_gen_params = QAConfigurations.from_dict(
        label_dataset_gen_params, api_keys
    )

    # get list of all documents in document_store_path
    document_store = glob.glob(f"{document_store_path}/*")

    with ChromaClient() as client:
        # Filter collections specific to the user_id.
        user_collections = [
            collection
            for collection in client.list_collections()
            if collection.name.startswith(f"userid_{user_id}_")
        ]

        # Check if there are any collections with the QA identifier and delete them.
        if any(["_qaid_0_" in collection.name for collection in user_collections]):
            for collection in user_collections:
                client.delete_collection(name=collection.name)

    # start generation process
    await agenerate_and_save_dataset(
        label_dataset_gen_params, document_store, label_dataset_path, user_id
    )
