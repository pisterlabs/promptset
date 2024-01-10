import openai
from openai.embeddings_utils import cosine_similarity
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
from EDGARFilingUtils import ROOT_DATA_DIR, filter_chunks, split_text, TICKER_TO_COMPANY_NAME, QUESTION_TO_CATEGORY 

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from time import sleep

import logging
logging.getLogger().setLevel(logging.ERROR)

import re
SECTION_DELIM_PATTERN = re.compile("####.+") # for pooled 10k files
EMBEDDING_CACHE_DIR = ROOT_DATA_DIR / "embedding_cache"

# For the purposes of measuring relevance between long docs and short queries,
# We want to use the text-search doc and query embeddings 
# (https://beta.openai.com/docs/guides/embeddings/what-are-embeddings)
# Keep these here so that we can simply refer to the model family when using these wrappers.
EMBEDDING_MODELS = {"ada": {"query": "text-search-ada-query-001",
                            "doc": "text-search-ada-doc-001",
                            "completion": "text-ada-001"
                            },
                    "babbage": {"query": "text-search-babbage-query-001",
                                "doc": "text-search-babbage-doc-001",
                                "completion": "text-babbage-001"
                                },
                    "curie": {"query": "text-search-curie-query-001",
                              "doc": "text-search-curie-doc-001",
                              "completion": "text-curie-001"
                              },
                    "davinci": {"query": "text-search-curie-query-001",
                                "doc": "text-search-curie-doc-001",
                                "completion": "text-davinci-002"
                                }
                    }

def call_openai_api_completion(prompt, model_family='ada',temperature=0.0):
    """Send a request to OpenAI's text generation API endpoint,
    with send_prompt and model.

    Args:
        prompt (str): The full prompt. 
        model_family (str, optional): model family to use for generation. Can be any of "ada", "babbage", "curie", "davinci". 
        Defaults to 'ada'.
        temperature (float): The temperature of the model. Range from 0 to 1. 
        0 will only pick most probably completions, while 1 selects lower probability completions. Default 0.

    Returns:
        str: The top scoring autocompletion. 
    """

    response = openai.Completion.create(
      model=EMBEDDING_MODELS[model_family]["completion"],
      prompt=prompt,
      max_tokens=400,
      temperature=temperature,
      stop=[".\n\n"]
    )
    return response['choices'][0]

@retry(wait=wait_random_exponential(min=6, max=30), stop=stop_after_attempt(10))
def get_embedding(text, model):
    """Given a string of long-form text, produce the embedding using the corresponding text-search-doc API endpoint.

    Args:
        text (str): String to produce an embedding for.
        model_family (str, optional): OpenAI model family to use text-search-doc for. Can be any of "ada", "babbage", "curie", "davinci".
        Defaults to "babbage".

    Returns:
        np.ndarray: Vector representation of the text.
    """
    embedding = None
    try:
        embedding = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
    except Exception as e:
        raise e
    return embedding

def query_similarity_search(embeddings, query, model_family="babbage", n=3, min_similarity=0.0, pprint=True):
    """Search the doc embeddings for the most similar matches with the query.

    Args:
        embeddings (DataFrame): df containing 'text' field, and its search/doc embeddings.
        query (str): Question to embed.  Uses the 'query' version of the embedding model.
        model_family (str, optional): model name.  can be "davinci", "curie", "babbage", "ada"; Default "babbage"
        n (int, optional): number of top results. Defaults to 3.
        pprint (bool, optional): Whether to print the text and scores of the top results. Defaults to True.

    Returns:
       DataFrame: Top n rows of the embeddings DataFrame, with similarity column added. Sorted by similarity score from highest to lowest. 
    """
    embedded = get_embedding(query, EMBEDDING_MODELS[model_family]['query'])
    embeddings["similarities"] = embeddings["doc_embeddings"].apply(lambda x: cosine_similarity(x, embedded))

    res = embeddings.sort_values("similarities", ascending=False).head(n)
    if pprint:
        print(f"{'-'*50}\nQuery: {query}\n{'-'*50}")
        for _, series in res.iterrows():
            if float(series["similarities"]) > min_similarity:
                print(f"Score: {series['similarities']:.3f}")
                print(series["text"])
                print()
    return res

def questions_to_answers(list_of_questions,embeddings,answers_per_question=5, min_similarity=0.0, model_family="curie",pprint=True):

    question_results = []
    for question in list_of_questions:
        top_similar = query_similarity_search(embeddings=embeddings,
                                              query=question,
                                              model_family=model_family,
                                              n=answers_per_question,
                                              min_similarity=min_similarity,
                                              pprint=pprint)
        top_similar["Question"]=question
        question_results.append(top_similar.drop(columns=["n_tokens","doc_embeddings"]))

    return pd.concat(question_results)

def query_to_summaries(list_of_query_questions, completion_temperature = 0.5,print_responses=True):
    """Given a list of search queries, embed them, and search the chunk database for most similar response.
    Then prompt GPT-3 to summarize the resulting sections. 

    Args:
        list_of_query_questions (list(str)): list of question strings to embed, searching for similar document chunks. 
        completion_temperature (float, optional): Temperature for davinci.
        print_responses (boolean, optional): whether to print the results to terminal. Default True.

    Returns:
        pd.DataFrame('filename', 'query', 'response',"confidence"): DataFrame containing the filename, query, and completion.
    """
    questions_to_gpt3_completions = []
    for industry in ["4_food_bev","11_transportation"]:
        for fname in glob(f"data/ind_lists/{industry}/10k/*_pooled.txt"):
            embeddings = file_to_embeddings(Path(fname),use_cache=True)
            df_questions_to_relevant_passages = questions_to_answers(list_of_query_questions,
                                                                     embeddings,
                                                                     answers_per_question=1,
                                                                     min_similarity=0.25,
                                                                     model_family='curie',pprint=False)
            for _, fields in df_questions_to_relevant_passages.iterrows():
                completion_prompt = produce_prompt(fields["text"],"") 
                completion_resp =call_openai_api_completion(completion_prompt,model_family="davinci",temperature=completion_temperature) 
                questions_to_gpt3_completions.append((Path(fname).stem,fields["Question"],completion_resp["text"],fields["similarities"]))
    if print_responses:
        for (fname, question, gpt3_completion,confidence) in questions_to_gpt3_completions:
                print("For filing", fname)
                print("For Question:")
                print(question,"\n")
                print(f"GPT-3 Responds with confidence {confidence}:")
                print(gpt3_completion)
    # Refactor the response to front end standard


    return pd.DataFrame(data=questions_to_gpt3_completions,columns=["filename","query","response","confidence"]) 

def change_completions_to_frontend_format(df_completions):

    response_jsons = {}
    for key, subdf in df_completions.groupby("filename"):
        print(key)
        company_ticker=re.match("[a-zA-Z]+",key).group()
        company_name=TICKER_TO_COMPANY_NAME[company_ticker]
        company_json = {"name":company_name}
        qa_pairs = []
        for question, subsubdf in subdf.groupby("query"):
            qrc_dict = {"question":question,
                                            "category": QUESTION_TO_CATEGORY[question]}
            answers = []
            for _, answer in subsubdf.iterrows():
                answer_dict = {"answer": answer["response"],"confidence":answer["confidence"]}
                answers.append(answer_dict)
            qrc_dict["answers"] = answers
            qa_pairs.append(qrc_dict)
        company_json["qa_pairs"] = qa_pairs
        response_jsons[company_name] = company_json
    

    return response_jsons
def file_to_embeddings(filepath, text_chunks = None, use_cache=True):
    """Given a filepath, produce a DataFrame containing the filtered text chunks, with their embeddings and number of tokens,
    if the DataFrame isn't cached. If it saved to disk, just load the DataFrame.

    Args:
        filename (Path): Pathlib.Path repr of the filepath of the file to be chunked and embedded.
        text_chunks (list(str), optional): list of chunked text, if already parsed. 
        use_cache (boolean,optional): Whether to load the DataFrame from disk or produce a new one and overwrite. 

    Returns:
        DataFrame: DataFrame containing columns "text", "n_tokens", "doc_embedding". Each entry corresponds to one chunk of the text.
    """

    if not EMBEDDING_CACHE_DIR.exists():
        EMBEDDING_CACHE_DIR.mkdir()
    # Search for the pickle, and read it in if it exists and use_cache is True.
    pickle_path = EMBEDDING_CACHE_DIR / f"{str(filepath.name).replace('.','_')}_embeddings.pkl" 
    if pickle_path.is_file()  and use_cache:
        return pd.read_pickle(str(pickle_path))
    
    # Read in and parse the file, if not passed in.
    if not text_chunks:
        raw_text = filepath.read_text(encoding="utf-8").replace("$","\$")
        if "pooled" in str(filepath): # pooled 10-K files are split into item1, item1a, item7 using a delimiter. 
            items = re.split(SECTION_DELIM_PATTERN,raw_text)
            text_chunks = []
            for item in items:
                section_chunked = split_text(item,form_type="10KItemsOnly")
                for chunk in section_chunked:
                    text_chunks.append(chunk)
        else:
            text_chunks = filter_chunks(split_text(raw_text))

    embeddings = []
    for i, text in enumerate(text_chunks):
        embedding_row = {}
        embedding_row["text"] = text
        embedding_row["n_tokens"] = len(tokenizer.encode(text))
        embedding_row["doc_embeddings"] = get_embedding(text, EMBEDDING_MODELS["curie"]["doc"])
        embeddings.append(embedding_row) 
        if (i+1)%10 == 0:
            print(f"{i+1} Chunks embedded.")
    df_embeddings = pd.DataFrame(embeddings)


    df_embeddings.to_pickle(str(pickle_path))

    return df_embeddings


def produce_prompt(context, query_text):
    """Produce the prompt by appending the query text with the context.

    Args:
        context (str): Context to try to answer the question with.
        query_text (str): Question to ask.

    Returns:
        str: Prompt to prime GPT-3 completion API endpoint.
    """

    return f"Provide a two sentence summary of the following text snippets:\n\nText Snippets:\n{context}\n\nAnswers:\n" 
    return f"""From the 10-K excerpt below:\n\n{context}\n\nCan you paraphrase an answer to the following question: {query_text}\n\nAnswer:"""