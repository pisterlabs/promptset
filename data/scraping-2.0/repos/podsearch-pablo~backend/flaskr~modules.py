import openai
import os
import re
from typing import Set
from transformers import GPT2TokenizerFast
import json
import numpy as np
import pandas as pd
import time
import sys

import csv
from nltk.tokenize import sent_tokenize
import ast





tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
COMPLETIONS_MODEL = "text-davinci-003"
openai.api_key = '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'


MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

MAX_SECTION_LEN = 1000
SEPARATOR = "\n* "

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}


def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def reduce_long(
    long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
    """
    Reduce a long text to a maximum of `max_len` tokens by potentially cutting at a sentence end
    """
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i][:-1]) + "."
    return long_text



def loadTimeStampSegments():
    """
    Returns the time stamp segments loaded from 'timestampsegments.json' as a dictionary
    """

    totalText = {}
    with open('./flaskr/timestampsegments.json') as fp:
        totalText = json.load(fp)

    sample = []
    for i in range(1,70):
        if i == 3:
            continue
        if (len(totalText[str(i)])!=0):
            sample.append(totalText[str(i)])

    count = 0
    df = []
    for i in range(len(sample)):
        tempCount = 0
        category = i
        for a in sample[i].keys():
            lst = []
            lst.append(category)

            if (count_tokens(sample[i][a])>1250):
                sample[i][a] = sample[i][a][:4000]
            tempCount+=len(sample[i][a].split())
            lst.append(a)
            lst.append(sample[i][a])
            lst.append(count_tokens(sample[i][a]))
            df.append(lst)

        count+=tempCount
    data = pd.DataFrame(df, columns=['title', 'heading', 'content', 'tokens'])
    return data


def get_embedding(text: str, model: str) -> list[float]:
    """
    Creates and returns an embedding from passed in string and OpenAI model
    """

    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    """
    Returns the embeddings for a given text, using pretermined document embedding model
    """
    time.sleep(1)
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> list[float]:
    """
    Returns the embeddings for a given text, using pretermined query embedding model
    """
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()
    }

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference. 
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Creates a prompt given a question, context embeddings, and a dataframe of TODO
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
         
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    most_relevant_document_sections = most_relevant_document_sections[:3]
    return (header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:", most_relevant_document_sections)

def construct_prompt_poem(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Creates a prompt (which returns a poem) given a question, context embeddings, and a dataframe of TODO
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
         
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, in the format of a poem, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    print(header)
    most_relevant_document_sections = most_relevant_document_sections[:3]
    return (header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:", most_relevant_document_sections)


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    """
    Answers a passed in query based on passed in embeddings and dataframe of TODO
    """ 
    
    (prompt, values) = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    return (response["choices"][0]["text"].strip(" \n"), values)

def answer_query_with_context_poem(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    """
    Answers a passed in query based on passed in embeddings and dataframe of TODO in poem format
    """ 
    
    (prompt, values) = construct_prompt_poem(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    

    return (response["choices"][0]["text"].strip(" "), values)


def load_context_embeddings():
    """
    Reads context embeddings from context_embeddings.csv
    """
    dct = pd.read_csv('./flaskr/context_embeddings.csv')
    lst = {}
    dct = dct.values.tolist()
    for i in range(1, len(dct)):
        lst[i] = ast.literal_eval(dct[i-1][1])
    return lst


def answer_question(text):
    """
    Answers a passed in question using loaded embeddings + predetermined models
    """

    new_context = load_context_embeddings()
    df = loadTimeStampSegments()
    (answer, relevant) =  (answer_query_with_context(text, df, new_context))
    print(answer)
    newRelevant = []
    for i in range(len(relevant)):
        newRelevant.append(getTupleFromIndex(relevant[i][1]))
    return {
        "answer": answer,
        "relevant": newRelevant
    }


def answer_question_poem(text):
    """
    Answers a passed in question using loaded embeddings + predetermined models, in the format of a poem
    """
    new_context = load_context_embeddings()
    df = loadTimeStampSegments()
    (answer, relevant) =  (answer_query_with_context_poem(text, df, new_context))
    print("ANSWER IS " + str(answer))
    newRelevant = []
    for i in range(len(relevant)):
        newRelevant.append(getTupleFromIndex(relevant[i][1]))
    return {
        "answer": answer,
        "relevant": newRelevant
    }


def getTupleFromIndex(index):
    """
    Given an index of a clip, returns the relevant timestamp, name of video, and link
    """

    with open('./flaskr/timestampsegments.json') as info:
        dct = json.load(info)
    with open ('./flaskr/timestamps.json') as timest:
        timeStamps = json.load(timest)
    with open ('./flaskr/youtubeVids.json') as videos:
        vids = json.load(videos)
    
    timeDct = {}
    a = 0
    for key in dct.keys():
        for key_key in dct[key].keys():
            timeDct[a]=(key, key_key)
            a+=1


    dctWhichContains = timeStamps[timeDct[index][0]]
    valueToFind = timeDct[index][1]
    
    name = (vids[str(timeDct[index][0])]['Name'])
    link = (vids[str(timeDct[index][0])]['Link'])
    
    time = (list(dctWhichContains.keys())[list(dctWhichContains.values()).index(valueToFind)]) 
    
    return (time, valueToFind, name, link, timeDct[index][0])


def main():
    answer_question("Why is dream a purple cow?")


 
if __name__ == "__main__":
    main()
