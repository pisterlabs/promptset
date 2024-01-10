import azure.functions as func
import logging
import requests
import json

from FlagEmbedding import FlagModel
from qdrant_client import QdrantClient

from config import EMBAAS_API_KEY, QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY


def get_embedding_embaas(text):
    """
    Get the embedding for the text using embaas and BGE model
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': EMBAAS_API_KEY
    }
    data = {'texts': [text], 'model': 'bge-large-en-v1.5'}
    response = requests.post('https://api.embaas.io/v1/embeddings/', json=data, headers=headers)
    embeddings = [entry['embedding'] for entry in response.json()['data']]
    return embeddings[0]


def get_embedding(query):
    """
    Get the query embedding using the Flag model local version
    """
    query_instruction_for_retrevial = ('Generate a representation for this podcast excerpt that can be used to '
                                       'find what the podcast is about?')
    model = FlagModel('BAAI/bge-large-en-v1.5', query_instruction_for_retrieval=query_instruction_for_retrevial,
                      use_fp16=True)
    embedding = model.encode(query).tolist()
    return embedding


def get_client(mode='local'):
    """
    Get the qdrant client
    @param mode: Mode to run in. Can be 'local' or 'remote'
    @return: Qdrant client
    """
    if mode == 'local':
        qdrant_client = QdrantClient("localhost", port=6333)
    else:  # if mode == 'remote'
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    return qdrant_client


def get_qdrant_results(embedding, n, collection_name):
    client = get_client()
    search_results = client.search(
        collection_name=collection_name, query_vector=embedding, limit=n
    )

    # This is the result string that will be returned in json format
    # We will send the title, link and the snippet of the article as the result
    results = []
    for i, search_result in enumerate(search_results):
        result_dict = {
            'episode': search_result.payload['episode'],
            'title': search_result.payload['title'],
            'link': search_result.payload['link'],
            'text': search_result.payload['text']
        }
        results.append(result_dict)

    return results


def run_embedding(question, n, collection_name):
    # Get the embedding for the question using embaas
    embedding = get_embedding(question)
    results = get_qdrant_results(embedding, n, collection_name)
    return results


app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.route(route="embedfn")
def embedfn(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    question = req.params.get('question')
    nitems = req.params.get('n')
    collection = req.params.get('collection')

    if question and nitems and collection:
        results = run_embedding(question, nitems, collection)
        # Convert the list of dictionaries to a json string
        result_str = json.dumps(results, indent=2)
        return func.HttpResponse(result_str)
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
            status_code=200
        )


import os
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.schema import Document, StrOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate


def get_llm():
    # Get the LLM to use with LangChain

    # Local LLM using Ollama and Langchain
    # llm = Ollama(model="llama2:13b")

    # OpenAI LLM
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

    return llm


def get_references(data):
    """
    Get all the episodes numbers and then return a string with the title and link to the episode
    """

    # Rmove all that do not have an episode number
    data = [x for x in data if 'episode' in x]

    # Create a dictionary with the episode number as key and the value as the title, link and the number of times it is
    # in the list
    episodes = {}
    for x in data:
        if x['episode'] in episodes:
            episodes[x['episode']]['count'] += 1
        else:
            episodes[x['episode']] = {'title': x['title'], 'link': x['link'], 'count': 1}

    references = '<br><br><b>References:</b><br>'

    # Iterate over the dictionary and create a string with the title and link to the episode
    for key, value in episodes.items():
        references += f'\nEpisode {key}: <a href="{value["link"]}">{value["title"]}</a><br>'

    return references


def answer_chain(question, retrieved_results):
    llm = get_llm()

    # Prompt template for the question
    template = """Answer the question based only on the following transcript text. 

        Text: "{context}"

        Question: "{question}"
        """
    template = template.replace('{question}', f'{question}')
    prompt = PromptTemplate.from_template(template)

    # Create the chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define the stuff chain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='context')

    docs = [Document(page_content=x['text'], metadata={"source": "local"}) for x in retrieved_results]
    response = stuff_chain.run(input_documents=docs)

    references = get_references(retrieved_results)

    return response + '\n\n' + references


def run_question_answering(question):
    # Get the results from qdrant embedding search
    results = run_embedding(question, 5, 'twiml_ai_podcast')

    # Get the answer from the LLM
    response = answer_chain(question, results)
    return response


@app.route(route="ragfn", auth_level=func.AuthLevel.FUNCTION)
def ragfn(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    question = req.params.get('question')

    # name = req.params.get('name')
    # if not name:
    #     try:
    #         req_body = req.get_json()
    #     except ValueError:
    #         pass
    #     else:
    #         name = req_body.get('name')

    if question:
        result_str = run_question_answering(question)
        return func.HttpResponse(f"{result_str}")
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a question in the query string or in the request body for a personalized response.",
            status_code=200
        )


import sys


def summarize_chain(message_list, llm):
    """
    Given an array of messages, use the llm to summarize it to a single sentence.
    """
    prompt_template = """Write a concise single sentence summary of the following text:
    "{text}"
    """
    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='text')
    docs = [Document(page_content=msg, metadata={}) for msg in message_list]

    return stuff_chain.run(input_documents=docs)


def query_embedding_history(question, history, collection_name):
    # Summarize the history to a single sentence
    llm = get_llm()
    summarized_history = summarize_chain(history, llm)
    print('Summarized history: ', summarized_history)

    # Start the qdrant client
    client = get_client()

    # This is the query string. Turn into an embedding using the model
    embedding_question = f'{question} {summarized_history}'
    embedding = get_embedding(embedding_question)

    search_results = client.search(
        collection_name=collection_name, query_vector=embedding, limit=5
    )

    # Turn the answer into a list of dictionaries
    search_results = [x.payload for x in search_results]

    return search_results


def run_question_answering_with_history(question, history):
    # Retrieve the answers from the retrival model
    results = query_embedding_history(question, history, 'twiml_ai_podcast')

    history_results = [{'text': x} for x in history]

    # Add history to the beginning of the results
    results = history_results + results

    # Get the answer from the LLM
    response = answer_chain(question, results)

    return response


@app.route(route="ragfnhist", auth_level=func.AuthLevel.FUNCTION)
def ragfnhist(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get the question from the request as a json post
    data_str = req.get_body().decode('utf-8')

    # Decode it as a json
    data = json.loads(data_str)

    # CHeck if it has the keys question and history
    has_question_and_history = 'question' in data and 'history' in data

    if has_question_and_history:
        # Get the question and history from the json
        question = data['question']
        history = data['history']

        # If history is empty, then just run the normal question answering
        if len(history) == 0:
            return func.HttpResponse(run_question_answering(question))

        # Get the answer from the model
        response = run_question_answering_with_history(question, history)

        return func.HttpResponse(response)
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
            status_code=200
        )


# ----- Check the question --------------------------------------------

def ask_if_search_episodes(question):
    # Ask if this is a question about searching for podcast episodes about a topic
    template = """Answer the question based only on the following context. Please answer yes or no only.

    context:
    "
    {context}
    "
    
    Is this question asking about finding episodes about a topic?
    """

    # Get the LLM to use with LangChain
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(template)
    runnable = prompt | llm | StrOutputParser()
    answer = runnable.invoke({'context': question})

    # Lower case the answer
    answer = answer.lower()
    print(answer)
    # Check if has the word yes inside the string anywhere
    if 'yes' in answer:
        return True
    else:
        return False


def get_episode_title(episode_num):
    # Read the json file
    with open('episode_info.json', 'r') as f:
        info = json.load(f)

    # Check if the episode number is in th info dictionary
    if str(episode_num) not in info:
        return ''

    # Get the episode number
    episode_title = info[str(episode_num)]

    return episode_title


def ask_if_episode_num(question):
    """
    Ask LLM to classify the question
    1. Is this a question about a podcast episode of a specific number?
    """
    # Get the LLM to use with LangChain
    llm = get_llm()

    # Ask if this is a question about a podcast episode of a specific number
    template = """Answer the question based only on the following context. Please answer yes or no only.

    context:
    "
    {context}
    "    

    Is the question in the context asking about a specific episode number?
    """
    prompt = ChatPromptTemplate.from_template(template)
    runnable = prompt | llm | StrOutputParser()

    followup_template = """
    Answer the question based only on the following context. Please answer with a number only.

    context:
    "
    {context}
    "

    What is the episode number?
    """

    # Classify the question if this is about a specific episode
    answer = runnable.invoke({'context': question})
    # Lower case the answer
    answer = answer.lower()
    # print(answer)
    # Check if has the word yes inside the string anywhere
    if 'yes' in answer:
        # Ask for the episode number
        followup_prompt = ChatPromptTemplate.from_template(followup_template)
        followup_runnable = followup_prompt | llm | StrOutputParser()
        episode_number_str = followup_runnable.invoke({'context': question})
        # print(episode_number_str)
        # Remove everything except numbers
        episode_number = ''.join(filter(str.isdigit, episode_number_str))
        # specific_episode.append(int(episode_number))
        return episode_number
    else:
        return None


def summarize_history(history):
    summarized_history = ''
    if len(history) > 0:
        llm = get_llm()
        summarized_history = summarize_chain(history, llm)
    return summarized_history


def query_embedding_summarized_history(question, summarized_history, collection_name, limit=5):
    # Start the qdrant client
    client = get_client()

    # This is the query string. Turn into an embedding using the model
    embedding_question = f'{question} {summarized_history}'
    embedding = get_embedding(embedding_question)

    search_results = client.search(
        collection_name=collection_name, query_vector=embedding, limit=limit
    )

    # Turn the answer into a list of dictionaries
    search_results = [x.payload for x in search_results]

    return search_results


def run_question_answering_with_check(question, history=[]):
    # Check if this question is asking us to search through multiple episodes
    search_episodes = ask_if_search_episodes(question)

    # If this is a question about searching for episodes about a topic, we will search both the episode summary
    # embeddings and the full transcripts. We will also look for 7 results instead of 5.
    if search_episodes:
        print('Searching for episodes about a topic')
        # Retrieve the answers from the retrival model
        history_summary = summarize_history(history)
        results_transcripts = query_embedding_summarized_history(question, history_summary, 'twiml_ai_podcast')
        results_summary = query_embedding_summarized_history(question, history_summary, 'twiml_ai_podcast_summary',
                                                             limit=10)

        history_results = [{'text': x} for x in history]

        # Add history to the beginning of the results
        results = history_results + results_summary + results_transcripts

        # Get the answer from the LLM
        response = answer_chain(question, results)

        return response
    else:
        print('Not searching for episodes about a topic')

    # Check if this question is asking about a specific episode
    episode_num = ask_if_episode_num(question)

    # Check if episode_num is a number or not as well as not None
    if episode_num is not None and episode_num.isdigit():
        print('Searching for a specific episode')
        episode_title = get_episode_title(episode_num)

        question += f' {episode_title}'
    else:
        print('Not searching for a specific episode')

    # Retrieve the answers from the retrival model
    history_summary = summarize_history(history)
    results = query_embedding_summarized_history(question, history_summary, 'twiml_ai_podcast', limit=3)

    history_results = [{'text': x} for x in history]

    # Add history to the beginning of the results
    results = history_results + results

    # Get the answer from the LLM
    response = answer_chain(question, results)

    return response


@app.route(route="ragfncheck", auth_level=func.AuthLevel.FUNCTION)
def ragfncheck(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get the question from the request as a json post
    data_str = req.get_body().decode('utf-8')

    # Decode it as a json
    data = json.loads(data_str)

    # CHeck if it has the keys question and history
    has_question_and_history = 'question' in data and 'history' in data

    if has_question_and_history:
        # Get the question and history from the json
        question = data['question']
        history = data['history']

        # If history is empty, then just run the normal question answering
        if len(history) == 0:
            return func.HttpResponse(run_question_answering_with_check(question))

        # Get the answer from the model
        response = run_question_answering_with_check(question, history)

        return response
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
            status_code=200
        )
