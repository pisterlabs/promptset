from typing import Literal
from services.configuration import get_settings
import concurrent.futures

from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain


def create_llm(task: Literal["summarize"] | Literal["synthesis"], temperature: float = 0, max_tokens: int = 1024,):
    """
    Creates a language model for the given task. If the OPENAI_API_KEY is set, it will use the OpenAI API, otherwise it will use the HuggingFace Hub.

    Parameters
    - ``task``: The task to create the language model for. Can be either ``summarize`` or ``synthesis``.

    Returns
    ``llm``: The language model.

    Raises
    - ``ValueError``: If the task is not ``summarize`` or ``synthesis`` or the OPENAI_API_KEY or the HUGGINGFACEHUB_API_TOKEN are not set.
    """
    try:
        settings = get_settings()
        if settings.OPENAI_API_KEY != None:
            llm = ChatOpenAI(temperature=temperature,
                             openai_api_key=settings.OPENAI_API_KEY,
                             max_tokens=max_tokens,
                             model='gpt-3.5-turbo'
                             ) if task == "summarize" else ChatOpenAI(temperature=temperature,
                                                                      openai_api_key=settings.OPENAI_API_KEY,
                                                                      max_tokens=max_tokens,
                                                                      model='gpt-4',
                                                                      )
        else:
            llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct",
                                 huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN)
        return llm
    except ValueError as e:
        print(e)
        raise ValueError(
            "The task must be either 'summarize' or 'synthesis' and the OPENAI_API_KEY or the HUGGINGFACEHUB_API_TOKEN must be set.")


def create_map_summarization_chain(llm: BaseLanguageModel, prompt: str, input_variables: list[str]):
    """
    Creates a summarization chain for the given language model, prompt and input variables.

    Parameters 

    - ``llm``: The language model to use.
    - ``prompt``: The prompt string representation to use.
    - ``input_variables``: The input variables for the prompt to use.

    Returns 

    ``map_chain``: The summarization chain.

    Raises

    - ``Exception``: If the chain could not be created.
    """
    try:
        prompt_template = PromptTemplate(
            template=prompt, input_variables=input_variables)
        map_chain = load_summarize_chain(llm=llm,
                                         chain_type="stuff",
                                         prompt=prompt_template)
        return map_chain
    except Exception as e:
        raise Exception("Could not create summarization chain.")


async def summarize_docs(llm: BaseLanguageModel, prompt: str, input_variables: list[str], documents: list[Document]):
    """
    Summarizes the given documents independently using the given language model, prompt and input variables.

    Parameters
    - ``llm``: The language model to use.
    - ``prompt``: The prompt string representation to use must be aimed to summarize.
    - ``input_variables``: The input variables for the prompt to use.
    - ``documents``: The documents to summarize.

    Returns
    ``summary_list``: The list of summaries.

    Raises
    - ``Exception``: If any of the chain instances fail.
    """
    try:
        summary_list = []
        map_chain = create_map_summarization_chain(
            llm=llm, prompt=prompt, input_variables=input_variables)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_doc = {executor.submit(
                run_summarization, map_chain, doc): doc for doc in documents}

            for future in concurrent.futures.as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    chunk_summary = future.result()
                    summary_list.append(chunk_summary)
                except Exception as e:
                    print(
                        f"Error processing document: {doc.page_content}, error: {e}")

        return summary_list
    except Exception as e:
        raise Exception("Could not summarize documents.")


def run_summarization(map_chain: BaseCombineDocumentsChain, doc: Document):
    """
    Runs the summarization chain for the given document.

    Parameters

    - ``map_chain``: The summarization chain to use.
    - ``doc``: The document to summarize."""
    return map_chain.run([doc])


async def synthesis(llm: BaseLanguageModel, prompt: str, input_variables: list[str], summary_list: list[str]):
    """
    Synthesizes the given summaries using the given language model, prompt and input variables.

    Parameters
    - ``llm``: The language model to use.
    - ``prompt``: The prompt string representation to use must be aimed to synthesize.Must be aimed to synthesize.
    - ``input_variables``: The input variables for the prompt to use.
    - ``summary_list``: The summaries to synthesize.

    Returns
    ``output``: The output of the synthesis in plain text.

    Raises
    - ``Exception``: If the chain instance fails."""
    try:
        summaries = "\n".join(summary_list)
        summaries_doc = Document(page_content=summaries)
        reduce_chain = create_map_summarization_chain(llm=llm,
                                                      prompt=prompt,
                                                      input_variables=input_variables)

        output = await reduce_chain.arun([summaries_doc])
        return output
    except Exception as e:
        raise Exception("Could not synthesize summaries.")
