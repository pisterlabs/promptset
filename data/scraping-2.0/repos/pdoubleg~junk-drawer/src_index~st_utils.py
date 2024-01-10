import os
from typing import List, Tuple
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
from IPython.display import Markdown, display, HTML
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from llama_index.agent import ReActAgent
from llama_index.llms.base import ChatResponse
from llama_index.agent.react.types import BaseReasoningStep
import openai
import streamlit as st

from langchain.chat_models import ChatOpenAI

from auto_coder import ReActAgentWrapper


# Llama Index Data Agent
class ReActAgentWrapperReasoning(ReActAgentWrapper):
    """
    A wrapper class for the ReActAgent class that includes additional functionality for reasoning steps.

    Attributes:
        reasoning_steps_history (list): A list to store the history of reasoning steps.
        latest_reasoning_step (BaseReasoningStep): The latest reasoning step.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the ReActAgentWrapperReasoning with the given arguments and keyword arguments.
        Also initializes an empty list for reasoning_steps_history and sets latest_reasoning_step to None.
        """
        super().__init__(*args, **kwargs)
        self.reasoning_steps_history = []
        self.latest_reasoning_step = None

    def _process_actions(
        self, output: ChatResponse
    ) -> Tuple[List[BaseReasoningStep], bool]:
        """
        Processes the actions and updates the reasoning steps history and the latest reasoning step.

        Args:
            output (ChatResponse): The output from the chat.

        Returns:
            Tuple[List[BaseReasoningStep], bool]: A tuple containing a list of reasoning steps and a boolean indicating if the process is done.
        """
        reasoning_steps, is_done = super()._process_actions(output)
        self.reasoning_steps_history.append(reasoning_steps)
        self.latest_reasoning_step = reasoning_steps[-1] if reasoning_steps else None
        return reasoning_steps, is_done

    async def _aprocess_actions(
        self, output: ChatResponse
    ) -> Tuple[List[BaseReasoningStep], bool]:
        """
        Asynchronously processes the actions and updates the reasoning steps history and the latest reasoning step.

        Args:
            output (ChatResponse): The output from the chat.

        Returns:
            Tuple[List[BaseReasoningStep], bool]: A tuple containing a list of reasoning steps and a boolean indicating if the process is done.
        """
        reasoning_steps, is_done = await super()._aprocess_actions(output)
        self.reasoning_steps_history.append(reasoning_steps)
        self.latest_reasoning_step = reasoning_steps[-1] if reasoning_steps else None
        return reasoning_steps, is_done



# General purpose helper functions
def deduplicate(seq):
    """
    Source: https://stackoverflow.com/a/480227/1493011
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def add_month_year_to_df(df: pd.DataFrame, date_col_name: str) -> pd.DataFrame:
    """
    Create 'Jan 2023' formatted date. Used for printing citations.
    
    Example: df = add_month_year_to_df(df, 'coverage_date_completed')
    
    Args:
        df (pd.DataFrame): Dataframe containing a datetime column.
        date_col_name (str): The column containing the date.
    Returns:
        pd.DataFrame: Input df with a new column named 'datestamp'.
    """
    df['datestamp'] = df[date_col_name].dt.to_period('M')
    df['datestamp'] = df['datestamp'].dt.strftime('%b %Y')
    df['datestamp'] = df['datestamp'].astype('string')
    return df


def count_tokens(text: str) -> int:
    """Helper function to count tokens in a string"""
    # Note this is hard coded for OpenAI gpt models
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    text = str(text)
    return len(encoding.encode(text))


# Citation logic
def extract_citation_numbers_in_brackets(text: str) -> List[str]:
    """
    Captures citations from LLM response. 
    The target format is an integer inside of square brackets, i.e., [1].
    
    Example: citation_numbers = extract_citation_numbers_in_brackets(response)
    
    Args:
        text (str): LLM 'response' output string.
    Returns:
        List[str]: A list containing citations used by the LLM. 
    """
    matches = re.findall(r'\[(\d+)\]', text)
    return list(set(matches))


# Display helpers

def pretty_format_dict(d):
    return '\n'.join(f'{k}: {v}' for k, v in d.items())

def print_cited_sources(df: pd.DataFrame, citation_numbers: List[str]) -> None:
    """
    Display cited sources with active links.
    Note: This is only a temporary solution to test if we can get proper hyperlinks from LLM output.
    
    """
    for citation in citation_numbers:
        i = int(citation) - 1  # convert string to int and adjust for 0-indexing
        title = df.iloc[i]["llm_title"]
        link = f"{df.iloc[i]['full_link']}"
        # link = f"{df.iloc[i]['full_link']}"
        venue = df.iloc[i]["State"]
        date = df.iloc[i]["datestamp"]
        number = df.iloc[i]["index"]
        st.markdown(f"###### {[i+1]} [{title}]({link}) - {venue}, {date}, Number: {number}")
        
        
def print_keyword_tags(df: pd.DataFrame, citation_numbers: List[str], keyword_col_name: str) -> None:
    """Display top 3 keywords/phrases from each cited source"""
    keyword_list = []
    for citation in citation_numbers:
        i = int(citation) - 1  # convert string to int and adjust for 0-indexing
        kwrds = df.iloc[i][keyword_col_name]
        keyword_list.append(kwrds)
    distinct_keywords = deduplicate(keyword_list)
    top_3_keywords = [s.split(',')[:3] for s in distinct_keywords]
    keyword_str = ""
    for i in range(len(top_3_keywords)):
        text = ", ".join(top_3_keywords[i])
        keyword_str += text+", "
    st.markdown(f" **Keyword Tags:** {keyword_str}")


# Top n RAG pipeline steps
def get_llm_fact_pattern_summary(df: pd.DataFrame, text_col_name: str) -> pd.DataFrame:
    """
    Function for using an LLM to prep top n search results.
    Operates on one row at a time so that exceptions can be caught in a way that doesn't break the 'chain'
    
    Example: top_n_res_df = get_llm_fact_pattern_summary(df=top_n_res_df, text_col_name="situation_question_clean")
    
    Args:
        df (pd.DataFrame): Results from top n search.
        text_col_name (str): The column of text to process. Should already be pre-processed.
    Returns:
        pd.DataFrame: Input df with a new column named 'summary'.
    """
    
    llm_text_prep_system_template = "You are an actuarial documentation specialist. You're helping me summarize legal questions related to insurance."
    llm_text_prep_human_template = "Using the following scenario, write a comprehensive fact-based summary that focuses on insurance and legal topics.\n\nDESCRIPTIONS:{text}\n\nSUMMARY:"


    llm_text_prep_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(llm_text_prep_system_template),
            HumanMessagePromptTemplate.from_template(llm_text_prep_human_template),
        ],
        input_variables=["text"],
    )

    llm_text_prep_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=512), prompt=llm_text_prep_prompt, verbose=False
    )
    
    df = df.reset_index(drop=False)
    list_to_chain = df[[text_col_name]].to_dict('records')
    
    batch_size = 1
    results = []
    for i, batch_start in tqdm(enumerate(range(0, len(list_to_chain), batch_size)), total=len(list_to_chain)//batch_size):
        try:
            batch = list_to_chain[batch_start: batch_start + batch_size]
            processed_batch = llm_text_prep_chain.run(batch)
            results.append(processed_batch)
        except (openai.error.InvalidRequestError, ValueError) as e:
            results.append({'text': str(e)})
    assert len(results) == len(df), "Length of the result does not match the original data!"
    
    df['summary'] = results
    df['summary'] = df['summary'].astype('string')

    return df


# Prior version using Instructor model embeddings
# def rerank_instructor(query_embeddings_model,
#            doc_embeddings_model,
#            df: pd.DataFrame, 
#            query: str, 
#            text_col_name: str = "summary",
# ) -> pd.DataFrame:
#     """
#     This function takes in the results of a top n search and returns re-ranked df
#     Performs the following steps:
#       1. Embed the query (should already be pre-processed)
#       2. Embed each top n result (designed to take in the df output from `get_llm_fact_pattern_summary`
#       3. Calculate cosine similarity for query and doc embeddings
#       4. Sory by similarity
    
#     Example: rerank_res_df = rerank(top_n_res_df, query, 'summary')
    
#     Args:
#         df (pd.DataFrame): Results from `get_llm_fact_pattern_summary`.
#         text_col_name (str): The column of text to embed. Defaults to "summary". 
#     Returns:
#         pd.DataFrame: Input df sorted based on Instructor embeddings re-ranking.
#     """
#     query_embedding = query_embeddings_model.embed_query(query)
#     query_embedding = np.array(query_embedding)
#     query_embedding = np.expand_dims(query_embedding, axis=0)
#     doc_embeddings = doc_embeddings_model.embed_documents(df[text_col_name].tolist())
#     df["instruct_embeddings"] = list(doc_embeddings)
#     df["similarity"] = cosine_similarity(query_embedding, doc_embeddings).flatten()
#     # sort the dataframe by similarity
#     df.sort_values(by="similarity", ascending=False, inplace=True)
#     return df


def rerank_with_cross_encoder(df: pd.DataFrame,
                              query: str, 
                              text_col_name: str = "summary",
                              model_name: str = 'cross-encoder/ms-marco-MiniLM-L-2-v2',
    ) -> pd.DataFrame:
    """
    A function to rerank search results using pre-trained cross-encoder
    
    On models:
    Base models are listed below. More info here: https://www.sbert.net/docs/pretrained-models/ce-msmarco.html
    ms-marco-MiniLM-L-6-v2: performance
    ms-marco-MiniLM-L-2-v2: speed
    
    Example: rerank_res_df = rerank_with_cross_encoder(top_n_res_df, query, 'summary')
    
    Args:
        df (pd.DataFrame): Results from `get_llm_fact_pattern_summary`.
        text_col_name (str): The column of text to embed. Defaults to "summary". 
    Returns:
        pd.DataFrame: Input df sorted based on Instructor embeddings re-ranking.
    """
    # Initialize the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Prepare the data for the model
    query_df = pd.DataFrame({"query": [query]})
    query_fact_pattern_df = get_llm_fact_pattern_summary(df=query_df, text_col_name="query")
    query_proc = query_fact_pattern_df["summary"].iloc[0]  
    data = [(query_proc, text) for text in df[text_col_name].tolist()]
    # Tokenize the data
    features = tokenizer(*zip(*data), padding=True, truncation=True, return_tensors="pt")
    # Predict the scores
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
    # Convert scores to numpy array
    scores = scores.detach().numpy()
    # Add the scores to the dataframe
    df['scores'] = scores
    # Sort the dataframe by the scores in descending order
    df = df.sort_values(by='scores', ascending=False)
    return df



# Create context for LLM prompt
def create_context(df: pd.DataFrame, query: str, context_token_limit: int = 3000) -> str:
    """
    Function to create context for LLM prompt. 
    Designed to take in results from `rerank_with_cross_encoder`, and add max number of search results based on a specified token limit.
    
    Args:
        df (pd.DataFrame): Results from `rerank_with_cross_encoder`.
        query (str): The target test for the search. Should be pre-processed.
        token_limit (int): Max tokens to add to the context. 
            Defaults to 3k, which allows for ~1k output tokens (if using model with ~4k context).
    Returns:
        str: Formatted LLM context.
    """
    returns = []
    count = 1
    total_tokens = count_tokens(query)  # Start with tokens from the query
    # Add the text to the context until the context is too long
    for i, row in df.iterrows():
        # row['full_link'] = f"{row['full_link'].replace(' ', '')}"
        text = ("["
                + str(count)
                + "] "
                + row["summary"]
                + "\nURL: "
                + row["full_link"])
        text_tokens = count_tokens(text)
        if total_tokens + text_tokens > context_token_limit:
            break
        returns.append(text)
        total_tokens += text_tokens
        count += 1
    return "\n\n".join(returns)



# Construct the final prompt
def create_formatted_input(df: pd.DataFrame, query: str, context_token_limit: int = 3000,
    instructions: str ="""Instructions: Using the provided search results, write a detailed comparative analysis for a new query. ALWAYS cite results using [[number](URL)] notation after the reference. \n\nNew Query:""",
) -> str:
    """
    Creates formatted prompt combining top n search results (context), the target query, and final instructions for the LLM.
    
    Example: formatted_input = create_formatted_input(rerank_res_df, query)
    
    Args:
        df (pd.DataFrame): Results from `rerank`.
        query (str): The target test for the search. Should be pre-processed.
    Returns:
        str: Formatted LLM prompt.
    """

    context = create_context(df, query, context_token_limit)

    try:
        prompt = f"""{context}\n\n{instructions}\n{query}\n\nAnswer:"""
        return prompt
    except Exception as e:
        print(e)
        return ""
    


# Main chain - Adds system message and passes formatted prompt to LLM
def get_final_answer(formatted_input: str, llm) -> str:
    
    main_system_template = "You are helpful legal research assistant. Analyze the current legal question, and compare it to past cases. Using only the provided context, offer insights into how the researcher can use the past questions to address the new outstanding issue. Remember to use markdown links when citing the context, for example [[number](URL)]."
    main_human_template = "Context:\n\n{text}"

    main_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(main_system_template),
            HumanMessagePromptTemplate.from_template(main_human_template),
        ],
        input_variables=["text"],
    )
    
    answer_chain_gpt = LLMChain(
        llm=llm, prompt=main_prompt, verbose=False)
    
    response = answer_chain_gpt.run({"text": formatted_input})
    
    return response


def print_results(df: pd.DataFrame, query: str, response: str) -> None:
    """Helper that prints everything"""
    citation_numbers = extract_citation_numbers_in_brackets(response)
    # st.markdown(f"## New Query:")
    st.markdown(f"{query}")
    # st.markdown(f"## Model Response:")
    st.markdown(f"{response}")
    print_cited_sources(df, citation_numbers)
    # print_keyword_tags(df, citation_numbers, "key_phrases_mmr")
