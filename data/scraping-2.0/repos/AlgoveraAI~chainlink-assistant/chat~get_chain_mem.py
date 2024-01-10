from langchain.llms import base
import tiktoken
from chat.prompts_mem import (
    FINAL_ANSWER_PROMPT,
    FINAL_ANSWER_2_PROMPT,
    QUESTION_MODIFIER_PROMPT,
)
from chat.prompts_no_mem import (
    ROUTER_PROMPT,
)
from chat.utils import get_retriever_chain, get_streaming_chain
from utils import createLogHandler
from schemas import ChatResponse, Sender, MessageType

logger = createLogHandler(__name__, "logs.log")

model = "gpt-3.5-turbo"
try:
    encoding = tiktoken.encoding_for_model(model)
except KeyError:
    logger.error(f"Encoding for model {model} not found. Using default encoding.")
    encoding = tiktoken.get_encoding("cl100k_base")


def calculate_tokens(document, encoding):
    """Calculate the number of tokens in a list of documents."""
    return len(encoding.encode(document))


def concatenate_documents(documents, max_tokens):
    """Combine documents up to a certain token limit."""
    combined_docs = ""
    token_count = 0
    used_docs = []

    for doc in documents:
        doc_tokens = calculate_tokens(doc.page_content, encoding)
        if (token_count + doc_tokens) <= max_tokens:
            combined_docs += f"\n\n{doc.page_content}\nSource: {doc.metadata['source']}"
            token_count += doc_tokens
            used_docs.append(doc)

    return combined_docs, used_docs


def call_llm_final_answer(question, document, memory, chain, stream=False):
    """Call LLM with a question and a single document."""
    if stream:
        chain.prompt = FINAL_ANSWER_PROMPT
        return chain.apredict(
            question=question, document=document, history=memory.buffer
        )
    else:
        chain.prompt = FINAL_ANSWER_PROMPT
        return chain.predict(
            question=question, document=document, history=memory.buffer
        )


def call_llm_final_2_answer(question, document, memory, chain):
    """Call LLM with a question and a single document."""
    chain.prompt = FINAL_ANSWER_2_PROMPT
    return chain.apredict(question=question, document=document, history=memory.buffer)


async def process_documents(question, chain, memory, max_tokens=14_000):
    """Process a list of documents with LLM calls."""

    # Modify question if memory is not empty
    if memory.chat_memory.messages:
        logger.debug(f"Processing documents for question: {question}")
        chain.prompt = QUESTION_MODIFIER_PROMPT
        modified_question = chain.predict(question=question, history=memory.buffer)
        logger.debug(f"Modified question: {modified_question}")

    else:
        modified_question = question

    # Use router to get workkflow to use
    chain.prompt = ROUTER_PROMPT
    workflow = chain.predict(question=modified_question)
    logger.debug(f"Using workflow: {workflow}")

    # Get relevant documents
    documents = retriever.get_relevant_documents(modified_question, workflow=workflow)
    batches = []
    num_llm_calls = 0
    while documents:
        batch, used_docs = concatenate_documents(documents, max_tokens)
        batches.append(batch)
        # logger.info(f"Calling LLM with {batch}")
        documents = [doc for doc in documents if doc not in used_docs]
        num_llm_calls += 1
        logger.info(
            f"Num LLM call required: {num_llm_calls}. {len(documents)} documents remaining."
        )

    return batches, num_llm_calls, workflow


async def get_answer_memory(question, memory, max_tokens=14_000, manager=None):
    """Process documents and call LLM."""

    # Get retriever and chain
    retriever, base_chain = get_retriever_chain()

    resp = ChatResponse(
        sender=Sender.BOT, message="Retrieving Documents", type=MessageType.STATUS
    )
    await manager.broadcast(resp)

    # Main code that calls process_documents
    batches, num_llm_calls, workflow = await process_documents(
        question, chain=base_chain, memory=memory, max_tokens=max_tokens
    )

    # Get the stream chain
    chain_stream = get_streaming_chain(
        manager=manager, chain=base_chain, workflow=workflow
    )

    resp = ChatResponse(
        sender=Sender.BOT, message=f"Generating Answer", type=MessageType.STATUS
    )
    await manager.broadcast(resp)

    if num_llm_calls == 1:
        result = await call_llm_final_answer(
            question=question,
            document=batches[0],
            chain=chain_stream,
            stream=True,
            memory=memory,
        )
        return result, memory

    else:
        # Handle the list of batches
        results = []
        for batch in batches:
            result = call_llm_final_answer(
                question=question,
                document=batch,
                chain=base_chain,
                stream=False,
                memory=memory,
            )
            results.append(result)

        combined_result = " ".join(results)

        logger.info(f"Final LLM call with {len(results)} results.")
        combined_result = await call_llm_final_2_answer(
            question=question,
            document=combined_result,
            chain=chain_stream,
            memory=memory,
        )

        return combined_result, memory
