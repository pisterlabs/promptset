import numpy as np
import openai  # dependency problem: python -m pip install charset-normalizer==2.1.0
import pandas as pd
import re
import tiktoken

from langchain import PromptTemplate
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    """ Function that calls the OpenAI API with exponential backoff."""
    return openai.ChatCompletion.create(**kwargs)


def check_token_count(input_text: str,
                      model_name: str = "gpt-3.5-turbo-0613"):
    """ Function that counts tokens, to be used before API call.

    This function is used to count the number of tokens in the input text. It takes as input a text and a model name
    and returns the number of tokens in the text.

    Args:
        input_text(str): Input string to be searched for keywords
        model_name(str): name of the llm used (for OpenAI API Call)

    Returns:
        num_tokens(int): number of tokens in the input text

    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(input_text))

    return num_tokens
    

def extract_knowledge(keyword: str,
                      keyword_short: str,
                      template_name: str,
                      input_text: str,
                      model_name: str = "gpt-3.5-turbo-0613") -> str:
    """
    Function that extracts relevant value from text input, if present

    Args:
        keyword: keyword of interest
        keyword_short: abbreviated keyword
        input_text: result from fuzzy keyword search (relevant keyword matches + context words)
        model_name: name of the llm used (for OpenAI API Call)

    Returns:
        response_message: message content extracted from API response
        tokens_spent: tokens spent in the API call
    """
    # read in the prompt template
    prompt_template = PromptTemplate(
            input_variables=["keyword", "keyword_short", "input_text"],
            template=open(template_name).read())
    
    # set up the prompt message from prompt template
    prompt = prompt_template.format(keyword=keyword,
                                    keyword_short=keyword_short,
                                    input_text=input_text)
    
    # check token count
    token_count = check_token_count(prompt)

    # if message too long, skip
    if token_count > 4000:
        response_message = "Input message would exceed token count"
        tokens_spent = None

        return response_message, tokens_spent

    # insert prompt into message format
    messages = [{"role": "system", 
                 "content": "Wenn vorhanden, extrahiere relevante Zahlenwerte im folgenden Format = Keyword: Zahl Maßeinheit"},
                {"role": "user",
                 "content": prompt}]

    # api call
    try:
        response = chat_completion_with_backoff(
            model=model_name,
            messages=messages,
            max_tokens=40,
            n=1,
            stop=None,
            temperature=0,  # deterministic
            top_p=0,  # nucleus sampling
            frequency_penalty=-2,  # controlling repetitive responses
            presence_penalty=-2  # low = no need to avoid topics mentioned by user input
        )
    
    # if unsuccessful, save error as response
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        response_message = f"An error occurred: {e}"
        tokens_spent = response['usage']['total_tokens']

        return response_message, tokens_spent

    # if successful, extract response
    response_message = response["choices"][0]["message"]['content']
    tokens_spent = response['usage']['total_tokens']
    
    return response_message, tokens_spent


def extract_numerical_value(keyword_short: str,
                            input_df: pd.DataFrame) -> pd.DataFrame:
    """Function that extracts the numerical value from the extracted info,
    and stores it in a new column.

    Args:
        keyword_short: abbreviated keyword
        input_df: df with identifier and content column

    Returns:
        pd.DataFrame: including new column for the extracted value

    """
    # compile regex pattern incl. keyword
    regex_pattern = rf'{keyword_short}:\s*(\d+,\d+|\d+)\s*(?:[a-zA-Z]+\s*)?'
    compiled_pattern = re.compile(regex_pattern)

    input_df[f'{keyword_short}_extracted_value'] = input_df[f'{keyword_short}_agent_response'].str.extract(compiled_pattern)
    return input_df


def validate_value_occurrence(keyword_short: str,
                              input_df: pd.DataFrame) -> pd.DataFrame:
    """ Function that verifies whether the extracted value was actually contained in the input string.

    This function is used to validate whether the extracted value was actually contained in the input string.
    It takes as input a df and a keyword and returns a df with a validation column.

    Args:
        keyword_short: abbreviated keyword
        input_df: df with identifier, content and value column

    Returns:
        pd.DataFrame: including new column for validation

    """
    # True if NaN or input text did contain the same numerical value
    input_df['validation'] = input_df.apply(lambda row:
                                        pd.isna(row[f'{keyword_short}_extracted_value']) or
                                        (str(row[f'{keyword_short}_extracted_value']) in row[f'{keyword_short}_input']),
                                        axis=1)
    
    return input_df
   

def extract_knowledge_from_df(keyword_dict: dict,
                              input_df: pd.DataFrame,
                              id_column_name: str,
                              text_column_name: str,
                              model_name: str = "gpt-3.5-turbo-0613") -> pd.DataFrame:
    """ Function that extracts relevant value from text input, if present and validates it.

    This function is used to extract relevant information from a df of text snippets. It takes as input a df and a
    dictionary with keywords and returns a df with the extracted information and a validation column.

    Args:
        keyword_dict: dictionary containing keyword, keyword_short and template_name
        input_df: df
        id_column_name: name of the identifying column (e.g., filename)
        text_column_name: name of the column holding the relevant text
        model_name: name of the llm used (for OpenAI API Call)

    Returns:
        pd.DataFrame: containing input_text, extracted_value, validation

    """
    # extract variables from keyword_dict
    keyword = keyword_dict['keyword']
    keyword_short = keyword_dict['keyword_short']
    template_name = keyword_dict['template_name']

    # get input text column as a numpy array
    input_df = input_df[[id_column_name, text_column_name]]
    input_texts = input_df.iloc[:, 1].values
    
    # init arrays to store the results
    agent_responses = np.empty(len(input_texts), dtype=object)

    logger.info(f"Relevant keyword(s): {keyword}")
    logger.info(f"Extracting relevant info from text snippets via LLM agent.")
    
    # loop through input texts and extract agent responses
    for i, input_text in enumerate(input_texts):
        agent_response, _ = extract_knowledge(keyword=keyword,
                                              keyword_short=keyword_short,
                                              template_name=template_name,
                                              input_text=input_text,
                                              model_name=model_name)
        agent_responses[i] = agent_response
    
    # create df from the results
    all_responses = pd.DataFrame({
        'id': input_df[id_column_name],
        f'{keyword_short}_input': input_texts,
        f'{keyword_short}_agent_response': agent_responses
    })

    logger.info("Info extracted. Extracting numerical value from info.")

    # add column for extracted numerical value
    all_responses_final = extract_numerical_value(keyword_short=keyword_short,
                                                  input_df=all_responses)
    
    logger.info("Numerical values extracted. Validating their occurrence in input text.")

    # validate occurrence
    all_responses_final = validate_value_occurrence(keyword_short=keyword_short,
                                                    input_df=all_responses_final)

    # convert value column to float
    all_responses_final[f'{keyword_short}_extracted_value'] = all_responses_final[f'{keyword_short}_extracted_value'].str.replace(',', '.').astype(float)

    logger.info(f"Returning results for {keyword_short}.")

    return all_responses_final

