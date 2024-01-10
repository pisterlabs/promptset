# pylint: disable=no-name-in-module
# pylint: disable=no-self-argument
# pylint: disable=abstract-class-instantiated
# pylint: disable=arguments-differ
# pylint: disable=unused-argument
import locale
import logging
import os
import re
from datetime import datetime
from typing import List

import openai
import qdrant_client
import requests
import tiktoken
from django.conf import settings
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI, ChatVertexAI
from langchain.llms import OpenAI, VertexAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.vectorstores import Qdrant
from paradigm_client.remote_model import RemoteModel
from pydantic import BaseModel
from transformers import AutoTokenizer, pipeline

from ai.services.anonymize_service import Anonymizer
from ai.services.embedding import get_embedding, get_query_prefix
from documents.models import Category

logger = logging.getLogger("cassandre")


class DocsRetriever(BaseRetriever, BaseModel):
    """Simple BaseRetriever for qa chain"""

    documents: List[Document]

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self.documents

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError


class DocumentSearch:
    """
    This class is responsible for searching documents based on a given query.
    It uses the Qdrant client for similarity search and retrieves relevant documents.
    """

    def __init__(self, category):
        self.abbreviation_dict = {
            "sft": "supplément familial de traitement",
            "iff": "indemnité forfaitaire de formation",
        }
        self.category = category
        self.embeddings = get_embedding()
        self.query_prefix = get_query_prefix()
        url = settings.QDRANT_URL
        self.client = qdrant_client.QdrantClient(url=url, prefer_grpc=True)
        self.docsearch = Qdrant(
            self.client, self.category.slug, self.embeddings.embed_query
        )

    def get_relevant_documents(self, query, threshold=0.80, k=None):
        """
        This method retrieves the relevant documents based on the given query.
        It performs a similarity search using the Qdrant client and filters
        the results based on the threshold.

        Args:
            query (str): The search query.
            threshold (float, optional): The similarity score threshold. Defaults to 0.80.

        Returns:
            List[Document]: A list of relevant documents.
        """
        query = self.normalize_query(query)
        k = self.category.k if k is None else k
        res = self.docsearch.similarity_search_with_score(query, k=k)
        documents: List[Document] = []
        for doc, score in res:
            if score < threshold:
                continue
            page = doc.metadata.get("page", "")
            source = doc.metadata.get("origin", "")

            doc.page_content = f"{doc.page_content}\nsource: {source} - page {page}\n"
            documents.append(doc)
        return documents

    def hyde_query(self, query):
        """
        This method is a work in progress and not yet finished.
        """
        hypothetical_prompt_template = PromptTemplate(
            input_variables=["question"],
            template="""{{question}}""",
        )
        hypothetical_prompt = hypothetical_prompt_template.format(question=query)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "En tant que Cassandre, experte en mouvement inter-académique des enseignants, Cassandre, transforme la question en réponse hypothetique",
                },
                {"role": "user", "content": hypothetical_prompt},
            ],
        )

        return response["choices"][0]["message"]["content"]

    def normalize_query(self, query):
        """
        This method normalizes the query by converting it to lowercase
        and replacing abbreviations with their full forms.

        Args:
            query (str): The search query.

        Returns:
            str: The normalized query.
        """
        query = query.lower()
        for abbr, full_form in self.abbreviation_dict.items():
            query = re.sub(rf"\b{abbr}\b", f"{abbr} ({full_form})", query)
        return f"{self.query_prefix}{query}"


def search_documents(
    query,
    engine="gpt-3.5-turbo",
    category_slug="documents",
    prompt=None,
    k=None,
    callback=None,
):
    """
    This function searches for documents based on the provided query and category.

    Args:
        query (str): The search query.
        engine (str, optional): The search engine to use. Defaults to "gpt-3.5-turbo".
        category_slug (str): The category of the documents to search.
        callback (function, optional): A callback function to handle the search results.
            Defaults to None.

    Returns:
        list: A list of relevant documents.
    """
    query = Anonymizer().anonymize(query)
    category = Category.objects.get(slug=category_slug)

    document_search = DocumentSearch(category=category)
    documents = document_search.get_relevant_documents(query, k=k)

    engine_query_map = {
        "falcon": query_falcon,
        "mistral_instruct": query_mistral_instruct,
        "paradigm": query_lighton,
        "fastchat": query_fastchat,
        "vertexai": query_vertexai,
    }

    query_function = engine_query_map.get(engine, query_openai)
    prompt = category.prompt if prompt is None else prompt
    return query_function(prompt, query, documents, engine, callback=callback)


def query_lighton(prompt, query, documents, engine, callback):
    """
    This function queries the LightOn model for answers based on the provided query and documents.

    Args:
        prompt (str): The prompt.
        query (str): The search query.
        documents (list): The list of relevant documents.
        engine (str, optional): The search engine to use. Defaults to "gpt-3.5-turbo".
        callback (function, optional): A callback function to handle the search results.
            Defaults to None.

    Returns:
        dict: A dictionary containing the search results.
    """
    host_ip = os.environ["PARADIGM_HOST"]
    model = RemoteModel(host_ip, model_name="llm-mini")

    context = "###\n".join([doc.page_content for doc in documents])
    prompt_template = PromptTemplate(
        input_variables=["question", "context"], template=prompt
    )

    # Utilise l'API Tokenize pour obtenir les ID de tokens pour "Je ne sais pas"
    tokenize_response = model.tokenize("Je ne sais pas")

    # Récupére les ID de tokens à partir de la réponse
    token_ids = [list(token.values())[0] for token in tokenize_response.tokens]

    # Ajoute un biais positif pour ces tokens
    biases = {token_id: 5 for token_id in token_ids}

    prompt = prompt_template.format(context=context, question=query)

    logger.debug("### paradigm")
    logger.debug("Prompt: %s", prompt)
    logger.debug("Number of tokens: %d", len(model.tokenize(prompt).tokens))

    stop_words = [
        "\n\n",
        "\nQuestion:",
    ]  # List of stopping strings to use during the generation
    parameters = {
        "n_tokens": 200,
        "temperature": 0,
        "biases": biases,
        "stop_regex": r"(?i)(" + "|".join(re.escape(word) for word in stop_words) + ")",
    }

    paradigm_result = model.create(prompt, **parameters)
    if hasattr(paradigm_result, "completions") and len(paradigm_result.completions) > 0:
        return {"result": paradigm_result.completions[0].output_text}
    else:
        return {"result": "No completions found"}


def format_content(content):
    lines = content.split("\n")
    if lines:
        lines[0] = f"### {lines[0]}"
    return "\n".join(lines)


def query_openai(prompt, query, documents, engine, callback):
    """
    This function queries the OpenAI model with a
    given category, query, documents, engine, and callback.

    Args:
        prompt (str): The prompt.
        query (str): The search query.
        documents (list): The list of relevant documents.
        engine (str): The search engine to use.
        callback (function, optional): A callback function
            to handle the search results. Defaults to None.

    Returns:
        dict: A dictionary containing the search results.
    """
    # Set the locale to French
    locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")

    now = datetime.now()
    formatted_date_time = now.strftime("%d %B %Y à %H:%M")

    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template=f"Nous sommes le {formatted_date_time}\n{prompt}",
    )

    context = "\n".join([format_content(doc.page_content) for doc in documents])
    prompt = prompt_template.format(context=context, question=query)
    enc = tiktoken.get_encoding("cl100k_base")
    token_count = len(enc.encode(prompt))

    logger.debug("Prompt: %s", prompt)
    logger.debug("Number of tokens: %d", token_count)

    callbacks = [callback] if callback is not None else []

    max_tokens = 4096 - token_count

    if engine == "gpt-3.5-turbo-instruct":
        llm = OpenAI(
            streaming=True,
            temperature=0,
            model_name=engine,
            request_timeout=300,
            callbacks=callbacks,
            max_tokens=max_tokens,
        )
    else:
        llm = ChatOpenAI(
            streaming=True,
            temperature=0,
            model_name=engine,
            request_timeout=300,
            callbacks=callbacks,
        )

    question_answer = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=DocsRetriever(documents=documents),
    )
    question_answer.combine_documents_chain.llm_chain.prompt = prompt_template

    results = question_answer({"query": query}, return_only_outputs=True)
    return {"result": results["result"], "input": prompt, "token_count": token_count}
    # return results


def query_falcon(prompt, query, documents, engine, callback):
    """
    This function queries the Falcon model with
    a given category, query, documents, engine, and callback.

    Args:
        category (str): The category of the query.
        query (str): The search query.
        documents (list): The list of relevant documents.
        engine (str): The search engine to use.
        callback (function, optional): A callback function to handle the search results. Defaults to None.

    Returns:
        dict: A dictionary containing the search results.
    """
    now = datetime.now()
    formatted_date_time = now.strftime("%d %B %Y à %H:%M")

    prompt_template = PromptTemplate(
        input_variables=["question", "context"], template=f"{prompt}"
    )

    context = (
        "\n***\n" + "\n***\n".join([doc.page_content for doc in documents]) + "\n***\n"
    )
    prompt = prompt_template.format(context=context, question=query)
    logger.debug("Prompt: %s", prompt)

    data = {
        "system": "Only respond if the answer is contained in the text above",
        "messages": [prompt],
        "max_tokens": 500,
        "temperature": 0.2,
        "top_k": 10,
        "top_p": 0.5,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(settings.TEXT_SYNTH_API_KEY),
    }

    response = requests.post(
        "https://api.textsynth.com/v1/engines/falcon_40B-chat/chat",
        headers=headers,
        json=data,
    )
    response_json = response.json()
    return {"result": response_json["text"], "input": prompt}

def query_mistral_instruct(prompt, query, documents, engine, callback):
    """
    This function queries the Mistral instruct model with
    a given category, query, documents, engine, and callback.

    Args:
        category (str): The category of the query.
        query (str): The search query.
        documents (list): The list of relevant documents.
        engine (str): The search engine to use.
        callback (function, optional): A callback function to handle the search results. 
        Defaults to None.

    Returns:
        dict: A dictionary containing the search results.
    """
    now = datetime.now()
    formatted_date_time = now.strftime("%d %B %Y à %H:%M")

    prompt_template = PromptTemplate(
        input_variables=["question", "context"], template=f"{prompt}"
    )

    context = (
        "\n***\n" + "\n***\n".join([doc.page_content for doc in documents]) + "\n***\n"
    )
    prompt = prompt_template.format(context=context, question=query)
    logger.debug("Prompt: %s", prompt)

    data = {
        "system": "Only respond if the answer is contained in the text above",
        "messages": [prompt],
        "max_tokens": 500,
        "temperature": 0.2,
        "top_k": 10,
        "top_p": 0.5,
        "stop": "\nQ:"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(settings.TEXT_SYNTH_API_KEY),
    }

    response = requests.post(
        "https://api.textsynth.com/v1/engines/mistral_7B_instruct/chat",
        headers=headers,
        json=data,
    )
    response_json = response.json()
    return {"result": response_json["text"], "input": prompt}

def query_vertexai(prompt, query, documents, engine, callback):
    """
    This function queries the VertexAI model with a given category, query, documents, engine, and callback.

    Args:
        promt (str): The promt.
        query (str): The search query.
        documents (list): The list of relevant documents.
        engine (str): The search engine to use.
        callback (function, optional): A callback function to handle the search results. Defaults to None.

    Returns:
        dict: A dictionary containing the search results.
    """
    now = datetime.now()
    formatted_date_time = now.strftime("%d %B %Y à %H:%M")

    prompt_template = PromptTemplate(
        input_variables=["question", "context"], template=f"{prompt}"
    )

    context = (
        "\n***\n" + "\n***\n".join([doc.page_content for doc in documents]) + "\n***\n"
    )
    prompt = prompt_template.format(context=context, question=query)
    logger.debug("Prompt: %s", prompt)


    qa = RetrievalQA.from_chain_type(
        llm=ChatVertexAI(),
        chain_type="stuff",
        retriever=DocsRetriever(documents=documents),
    )
    qa.combine_documents_chain.llm_chain.prompt = prompt_template

    results = qa({"query": query}, return_only_outputs=True)
    return results


def query_fastchat(category, query, documents, engine, callback):
    """
    This function queries the FastChat model for answers based on the provided query and documents.

    Args:
        category (str): The category of the documents to search.
        query (str): The search query.
        documents (list): The list of relevant documents.
        engine (str, optional): The search engine to use. Defaults to "gpt-3.5-turbo".
        callback (function, optional): A callback function to handle the search results. Defaults to None.

    Returns:
        dict: A dictionary containing the search results.
    """
    # model = "tiiuae/falcon-7b-instruct"
    model = "lmsys/fastchat-t5-3b-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device="mps",
        model_kwargs={
            # "load_in_8bit": False,
            "max_length": 512,
            "temperature": 0.0,
        },
    )
    hf_llm = HuggingFacePipeline(pipeline=pipe)

    now = datetime.now()
    formatted_date_time = now.strftime("%d %B %Y à %H:%M")

    prompt_template = PromptTemplate(
        input_variables=["question", "context"], template=category.prompt
    )

    qa = RetrievalQA.from_chain_type(
        llm=hf_llm,
        chain_type="stuff",
        retriever=DocsRetriever(documents=documents),
    )
    qa.combine_documents_chain.llm_chain.prompt = prompt_template

    results = qa({"query": query}, return_only_outputs=True)
    return results
