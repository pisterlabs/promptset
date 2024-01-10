"""
Utility functions for working with LLMs.

Functions

set_openai_key() -> str
    Sets the OpenAI API key based on your environmental variables.
    Returns:
        A string representing the OpenAI API key.

get_models() -> None
    Prints a list of OpenAI models available to the user.

num_tokens(string: str, encoding_name: str = "cl100k_base") -> int
    Returns the number of tokens in a text string using the specified tokenizer.
    Parameters:
        string (str): The text string to tokenize.
        encoding_name (str): The name of the tokenizer to use. Defaults to "cl100k_base".
    Returns:
        An integer representing the number of tokens in the text string.

time_tokens(prompt_template: str, human_template: str, query: str, model: str = 'gpt-4') -> float
    Returns the time needed to wait before the next request to the language model.
    Parameters:
        prompt_template (str): The prompt template.
        human_template (str): The human template.
        query (str): The query.
        model (str): The model to use. Defaults to 'gpt-4'.
    Returns:
        A float representing the time to wait in seconds.

sleep_for_time_tokens(
    prompt_template: str, 
    human_template: str, 
    query: str, 
    model: str = 'gpt-4'
) -> None
    Sleeps for the time needed to wait before the next request to the language model.
    Parameters:
        prompt_template (str): The prompt template.
        human_template (str): The human template.
        query (str): The query.
        model (str): The model to use. Defaults to 'gpt-4'.

sleep_for_tokens(tokens: int, model: str = 'gpt-4') -> None
    Sleeps for the time needed to wait before the next request to the language model. 
    The number of tokens is provided as an argument.
    Parameters:
        tokens (int): The number of tokens.
        model (str): The model to use. Defaults to 'gpt-4'.

trim_to_last_blank_line(string: str) -> str
    Trims a string back to the last blank line.
    Parameters:
        string (str): The string to trim.
    Returns:
        A string trimmed back to the last blank line.

trim_for_tokens(string: str, max_tokens: int = 6000, max_attempts: int = 3000) -> str
    Trims a string to a maximum number of tokens.
    Parameters:
        string (str): The string to trim.
        max_tokens (int): The maximum number of tokens. Defaults to 6000.
        max_attempts (int): The maximum number of attempts to trim the string. Defaults to 3000.
    Returns:
        A string trimmed to the maximum number of tokens.

trim_part_for_tokens(
    part: str, remainder: str, 
    max_tokens: int = 6000, 
    trim_tokens: int = 3000, 
    max_attempts: int = 3000
) -> str
    Trims a part of a string so that the part plus a remainder is under a token limit.
    Parameters:
        part (str): The part of the string to trim.
        remainder (str): The remainder of the string.
        max_tokens (int): The maximum number of tokens. Defaults to 6000.
        trim_tokens (int): The number of tokens to trim. Defaults to 3000.
        max_attempts (int): The maximum number of attempts to trim the string. Defaults to 3000.
    Returns:
        A string trimmed to the maximum number of tokens.

string_to_token_list(string: str, chunk_size: int = 6000, chunk_overlap: int = 0) -> List[str]
    Turns a string into a list of token-sized strings.
    Parameters:
        string (str): The string to split into tokens.
        chunk_size (int): The size of each chunk. Defaults to 6000.
        chunk_overlap (int): The overlap between chunks. Defaults to 0.
    Returns:
        A list of strings, each of which is a token-sized chunk of the original string.

list_to_token_list(lst: List[str], chunk_size: int = 6000, chunk_overlap: int = 0) -> List[str]
    Combines strings in a list so that each string in the list approaches the token limit.
    Parameters:
        lst (List[str]): The list of strings to combine.
        chunk_size (int): The size of each chunk. Defaults to 6000.
        chunk_overlap (int): The overlap between chunks. Defaults to 0.
    Returns:
        A list of strings, each of which is a token-sized chunk of the combined strings.

list_to_db(
    lst: List[str],
    name: str = 'vectordb',
    path: str = get_root_dir() + '/ data / vectordb', settings: LLMSettings
) -> Chroma
    Creates a vector database based on a list of strings.
    Parameters:
        lst (List[str]): The list of strings to create the database from.
        name (str): Name of the database. Defaults to 'vectordb'.
        path (str): Path to the database. Defaults to the root directory plus '/ data / vectordb'.
        settings (LLMSettings): The settings for the database.
    Returns:
        A Chroma object representing the vector database.

load_db(path: str, embeddings: OpenAIEmbeddings = OpenAIEmbeddings()) -> Chroma
    Loads a vector database from a file.
    Parameters:
        path (str): The path to the database.
        embeddings (OpenAIEmbeddings): The embeddings to use. Defaults to OpenAIEmbeddings().
    Returns:
        A Chroma object representing the loaded vector database.

llm_call(prompt_template, human_template, query, model='gpt-4')
    Returns the output of an LLMChain call. 
    If the rate limit is reached, it will ask the user if they want to wait and retry.

llm_call_long(prompt_template, human_template, query, model='gpt-4-1106-preview')
    Returns the output of an LLMChain call with a longer token limit. 
    Defaults to using gpt-4-1106-preview, which has an enormous token limit.

llm_condense_string(string, prompt_condense, model='gpt-4', chunk_size=6000, chunk_overlap=200)
    Condenses the length of a string. 
    The string is broken up into a list of token-sized strings. An LLM is called to condense 
    each string in the list. The strings are then recombined as one string.

llm_condense_string_long(string, prompt_condense, settings=LLMSettings)
    Condenses the input of an LLM query using longer LLM.

llm_router(prompt_template, human_template, query, prompt_condense, settings=LLMSettings)
    Depending on the number of tokens of the input, runs different llm call functions.
    This function can handle inputs of any length.

llm_router_gpt4(prompt_template, human_template, query, prompt_condense, settings=LLMSettings)
    Depending on the number of tokens of the input, runs either llm_call or llm_condense. 
    This is designed to be the main function for calling GPT4 exclusively.

llm_loop(prompt_template, human_template, lst, prompt_condense, settings=LLMSettings)
    Loops through a list, calling llm_router on each item and sleeping for tokens.

llm_loop_gpt4(prompt_template, human_template, lst, prompt_condense, settings=LLMSettings)
    Loops through a list, calling llm_router_GPT4 on each item and sleeping for tokens.

"""
import logging
import os
import time
import textwrap
from dataclasses import dataclass
from dotenv import load_dotenv
import tiktoken
import openai
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    TokenTextSplitter
)
from langchain.vectorstores import Chroma
from src.utils_file import get_root_dir

# Set up logger
logger = logging.getLogger('restatement')

@dataclass
class LLMSettings:
    """Settings for the LLM."""
    # Embeddings
    embeddings = OpenAIEmbeddings()
    # Primary LLM model
    model: str = 'gpt-4'
    # Maximum tokens for input to primary LLM
    max_tokens: int = 6000
    # LLM model for long inputs
    model_long: str = 'gpt-4-1106-preview'
    # Maximum tokens for input to long LLM
    max_tokens_long: int = 40000
    # Chunk size for breaking up long inputs for primary LLM
    chunk_size: int = 4000
    # Overlapping text between chunks
    chunk_overlap: int = 200
    # Chunk size for breaking up long inputs for long LLM
    chunk_size_long: int = 40000
    # Maximum number of attempts at reducing a long input to a short input by breaking it up
    # into chunks, summarizing those chunks, and then combining the summaries.
    max_attempts: int = 3

def set_openai_key():
    """Set variable for OpenAI API key based on your environmental variables."""
    load_dotenv()
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    return openai_api_key

def get_models():
    """Prints a list of OpenAI models available to the user."""
    try:
        available_models = openai.models.list()

        for model in available_models.data:
            print(model.id)
    except Exception as e:
        print(f"Error: {e}")
        print("Unable to retrieve model information.")

def num_tokens(string, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string using the CL100k_base tokenizer."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = len(encoding.encode(string))
    return tokens

def time_tokens(prompt_template, human_template, query, model='gpt-4'):
    """Returns the time needed to wait before the next request to the LLM."""
    if model == 'gpt-3.5-turbo-1106':
        tps = 300000 / 60 # tokens per minute under my account
    else:
        tps = 3000000 / 60 # tokens per minute under my account
    tokens = num_tokens(prompt_template + human_template + query)
    return tokens / tps

def sleep_for_time_tokens(prompt_template, human_template, query, model='gpt-4'):
    """Sleeps for the time needed to wait before the next request to the LLM."""
    time_to_sleep = time_tokens(prompt_template, human_template, query, model)
    logger.debug("Sleeping for %s seconds...", time_to_sleep)
    time.sleep(time_to_sleep)

def sleep_for_tokens(tokens, model='gpt-4'):
    """Sleeps for the time needed to wait before the next request to the LLM.
    Difference between this and sleep_for_time_tokens is that tokens are provided in argument
    rather than text strings that the function converts to tokens.
    """
    if model == 'gpt-3.5-turbo-1106':
        tps = 300000 / 60 # tokens per minute under my account
    else:
        tps = 300000 / 60 # tokens per minute under my account
    time_to_sleep = tokens / tps
    logger.debug("Sleeping for %s seconds...", time_to_sleep)
    time.sleep(time_to_sleep)

def trim_to_last_blank_line(string):
    """Trims string back to the last blank line."""
    lines = string.splitlines()
    for i in range(len(lines)-1, -1, -1):
        if not lines[i].strip():
            return '\n'.join(lines[:i+1])
    return ''

def trim_for_tokens(string, max_tokens=6000, max_attempts=3000):
    """Trims string to max_tokens."""
    tokens = num_tokens(string)
    count = 1
    while tokens > max_tokens and count < max_attempts:
        logger.debug(
            "trim_for_tokens: String is too long. Trimming to last blank line. (Attempt %s).", 
            count
        )
        if string == trim_to_last_blank_line(string):
            logger.debug(
                "trim_for_tokens: String is too long. Trimming to last sentence. (Attempt %s).",
                count
            )
            string = string[:string.rfind(".")+1]
            tokens = num_tokens(string)
            count += 1
            continue
        string = trim_to_last_blank_line(string)
        tokens = num_tokens(string)
        count += 1
    return string

def trim_part_for_tokens(part, remainder, max_tokens=6000, trim_tokens=3000, max_attempts=3000):
    """Trims part so that part + remainder is under token limit."""
    if num_tokens(remainder) > max_tokens:
        logger.debug(
            "trim_query_for_tokens: Other parts exceed %s tokens. Reducing to %s tokens.",
            max_tokens,
            trim_tokens
        )
        part = trim_for_tokens(part, trim_tokens)
        return part
    tokens = num_tokens(part + remainder)
    count = 1
    while tokens > max_tokens and count < max_attempts:
        logger.debug(
            "trim_part_for_tokens: Part is too long. Trimming to last blank line. (Attempt %d).", 
            count
        )
        if part == trim_to_last_blank_line(part):
            logger.debug(
                "trim_part_for_tokens: Part is too long. Trimming to last sentence. (Attempt %d).", 
                count
            )
            part = part[:part.rfind(".")+1]
            tokens = num_tokens(part + remainder)
            count += 1
            continue
        part = trim_to_last_blank_line(part)
        tokens = num_tokens(part + remainder)
        count += 1
    return part

def string_to_token_list(string, chunk_size=6000, chunk_overlap=0):
    """Turns string into list of token-sized strings."""
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(string)

def list_to_token_list(lst, chunk_size=6000, chunk_overlap=0):
    """Combines strings in a list so that each string in the list approaches token limit.
    This increases efficiency when you want an LLM to process all the items in a list
    but you don't need to process each item individually with its own LLM call.
    """
    temp_list = lst
    token_list = []
    total_tokens = 0
    scratchpad = ""
    index = 0
    while index < len(temp_list):
        x = temp_list[index]
        try:
            tokens = num_tokens(x)
        except Exception as e:
            logging.error("Error calculating tokens for item at index %s: %s", index, e)
            index += 1
            continue
        # If item exceeds token limit, split item into its own token_list,
        # insert that list, end the current iteration, and continue to the next iteration.
        if tokens >= chunk_size:
            try:
                x_list = string_to_token_list(
                    x,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            except Exception as e:
                logging.error(
                    "Error splitting string into token list for item at index %d: %s", 
                    index,
                    e
                )
                index += 1
                continue
            temp_list.pop(index)
            temp_list[index:index] = x_list  # insert x_list items at current position
            continue
        # If item plus scratchpad exceeds token limit, add scratchpad to list,
        # clear scratchpad, and clear total tokens.
        if total_tokens + tokens >= chunk_size:
            token_list.append(scratchpad)
            scratchpad = ""
            total_tokens = 0
        # Add item to scratchpad, add tokens to token count
        scratchpad += "\n " + x
        total_tokens += tokens
        index += 1
    if scratchpad:  # handle any remaining content in scratchpad
        token_list.append(scratchpad)
    return token_list

def list_to_db(
        lst,
        name='vectordb',
        path=get_root_dir() + '/ data / vectordb',
        settings=LLMSettings
    ):
    """Create a vector database based on a list of strings.
    """
    # Create vector database.
    embeddings = settings.embeddings
    persist_directory = str(path) + f"/ {name}"
    persist_directory = str(persist_directory)
    vectordb = Chroma.from_texts(texts=lst,
                                 embedding=embeddings,
                                 persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

def load_db(
        path,
        embeddings = OpenAIEmbeddings()
    ):
    """Load a vector database from file.
    """
    vector_db = Chroma(persist_directory=path,embedding_function=embeddings)
    return vector_db

def llm_call(prompt_template, human_template, query, model='gpt-4'):
    """Returns the output of an LLMChain call.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        prompt_template
        )
    human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_template
        )
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
        )
    chat_prompt_str = textwrap.dedent(
    f"""
    System prompt:
    {prompt_template}
    Human prompt:
    {human_template}
    Query:
    {query}
    """
    )
    total_tokens = num_tokens(chat_prompt_str)
    while True:
        try:
            llm_chain = LLMChain(
                llm=ChatOpenAI(
                    model_name=model,
                    temperature=0.0,
                    verbose=False,
                    openai_api_key=set_openai_key(),
                    streaming=True
                    ),
                prompt=chat_prompt,
                verbose=False
                )
            output = llm_chain(query)
            logger.debug(output['text'])
            return (output, total_tokens, model, chat_prompt_str)

        except Exception as e:  # Replace Exception with the specific exception class, if known.
            print(f"Error: {e}")

            # Check if the error message corresponds to the limit being reached.
            if "rate limit" in str(e).lower():
                # Adjust the condition based on the actual error message.
                user_input = input("""llm_call: The LLM rate limit has been reached.
                                   Do you want to wait and retry? (yes/no): 
                                   """).lower()
                if user_input == 'yes':
                    while True:
                        try:
                            wait_time = int(input("How long (seconds)?: "))
                            break
                        except ValueError:
                            print("Please enter a valid number of seconds.")
                    print(f"Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)  # wait
                    continue
                else:
                    raise  # re-raise the exception if the user doesn't want to wait
            else:
                raise  # re-raise the exception if it's not related to the limit being reached

def llm_call_long(prompt_template, human_template, query, model='gpt-4-1106-preview'):
    """Returns the output of an LLMChain call with a longer token limit.
    Defaults to using gpt-4-1106-preview, which has an enormous token limit. The cost is
    that it performs worse than standard gpt-4 at complex tasks.
    """
    return llm_call(prompt_template, human_template, query, model=model)

def llm_condense_string(
        string,
        prompt_condense,
        model='gpt-4',
        chunk_size=6000,
        chunk_overlap=200
        ):
    """Condenses the length of a string.
    The string is broken up into a list of token-sized strings. An LLM is called to 
    condense each string in the list. The strings are then recombined as one string.
    """
    string_condensed = ""
    prompt_lst = []
    # Split up strings into token-sized list
    texts = string_to_token_list(string, chunk_size, chunk_overlap)
    logger.debug("llm_condense_string: Number of strings to condense: %s", len(texts))
    # Have LLM condense each string in the list. Then add each output to query_condensed.
    prompt_template = prompt_condense
    # Note that human template must containt {query} or LLMChain will error out.
    human_template = textwrap.dedent(
    """Please write a shorter version of this. 
    {query}
    """
    )
    count = 1
    for text in texts:
        output, total_tokens, model, chat_prompt_str = llm_call(
            prompt_template,
            human_template,
            text,
            model=model
        )
        # Add the output to the string_condensed.
        string_condensed += output["text"]+"\n"
        # Add the prompt to a list of prompts used in this function.
        prompt_lst.append(chat_prompt_str)
        logger.debug("llm_condense_string: String %s of %s condensed.", count, len(texts))
        # Sleep for tokens used.
        sleep_for_tokens(total_tokens, model)
        count += 1
    return (string_condensed, prompt_lst)

def llm_condense_string_long(
        string,
        prompt_condense,
        settings=LLMSettings,
    ):
    """Condenses the input of an LLM query using longer LLM."""
    return llm_condense_string(
        string,
        prompt_condense,
        model=settings.model_long,
        chunk_size=settings.chunk_size_long,
        chunk_overlap=settings.chunk_overlap
        )

def llm_router(
        prompt_template,
        human_template,
        query,
        prompt_condense,
        settings=LLMSettings
    ):
    """Depending on number of tokens of the input, runs different llm call functions. 
    This is the default function for calling a large language model. The token length of 
    the input does not need to be calculated in advance. This function can handle inputs
    of any length.
    Long inputs will be split into parts that an LLM can process. Each part will be condensed
    and then the parts will be recombined. The recombined text will then be processed by either 
    llm_call or llm_call_long.
    If the recombined text is still too long, the recombined text will be condensed and recombined
    up to two more times.
    Shorter inputs will be processed by either llm_call or llm_call_long, depending on 
    token length.
    """
    prompt_lst = []
    total_tokens = num_tokens(prompt_template + human_template + query)
    attempts = 0
    while total_tokens > settings.max_tokens_long and attempts < settings.max_attempts:
        logger.debug(
            "llm_router: Input is too long. Condensing... (Attempt %s/%s)", 
            attempts+1,
            settings.max_attempts
        )
        query, prompt = llm_condense_string_long(
            string=query,
            prompt_condense=prompt_condense,
            settings=settings
        )
        total_tokens = num_tokens(prompt_template + human_template + query)
        prompt_lst.extend(prompt)
        attempts += 1
    if total_tokens < settings.max_tokens:
        logger.debug("llm_router: Input is short enough for GPT4. Processing...")
        output, total_tokens, model, chat_prompt_str = llm_call(
            prompt_template,
            human_template,
            query,
            model=settings.model
        )
    elif total_tokens < settings.max_tokens_long:
        logger.debug("llm_router: Input is short enough for GPT3.5-turbo-1106. Processing...")
        output, total_tokens, model, chat_prompt_str = llm_call_long(
            prompt_template,
            human_template,
            query,
            model=settings.model_long
        )
    else:
        raise ValueError(
            "After "+str(settings.max_attempts)+" attempts, the input is still too long."
        )
    prompt_lst.append(chat_prompt_str)
    return (output, total_tokens, model, prompt_lst)

def llm_router_gpt4(
        prompt_template,
        human_template,
        query,
        prompt_condense,
        settings=LLMSettings
    ):
    """Depending on number of tokens of the input, runs either llm_call or llm_condense.
    This is designed to be the main function for calling GPT4 exclusively,
    because the token length of the input does not need to be calculated in advance.
    If the input is within GPT4's token limit, it will be processed by llm_call.
    Else it will be condensed by llm_condense (until it is short enough) and then 
    processed by llm_call.
    """
    prompt_lst = []
    total_tokens = num_tokens(prompt_template + human_template + query)

    attempts = 0
    while total_tokens > settings.max_tokens and attempts < settings.max_attempts:
        logger.debug(
            "llm_router_gpt4: Input is too long. Condensing... (Attempt %s/%s)", 
            attempts+1,
            settings.max_attempts
        )
        query, prompt = llm_condense_string(
            query,
            prompt_condense,
            model=settings.model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        total_tokens = num_tokens(prompt_template + human_template + query)
        prompt_lst.extend(prompt)
        attempts += 1
    if total_tokens <= settings.max_tokens:
        logger.debug("llm_router_gpt4: Input is short enough for GPT4. Processing...")
        output, total_tokens, model, chat_prompt_str = llm_call(
            prompt_template,
            human_template,
            query,
            model=settings.model
        )
        prompt_lst.append(chat_prompt_str)
        return (output, total_tokens, model, prompt_lst)
    else:
        raise ValueError(f"After {settings.max_attempts} attempts, the input is still too long.")

def llm_loop(
        prompt_template,
        human_template,
        lst,
        prompt_condense,
        settings=LLMSettings
    ):
    """Loops through list, calling llm_router on each item and sleeping for tokens."""
    output_lst = []
    prompt_lst = []
    count = 1
    for item in lst:
        logger.debug("llm_loop: Processing item %s of %s", count, len(lst))
        output, total_tokens, model, prompt = llm_router(
            prompt_template,
            human_template,
            item,
            prompt_condense,
            settings=settings
        )
        output_lst.append(output["text"])
        prompt_lst.extend(prompt)
        count += 1
        sleep_for_tokens(total_tokens, model)
    return output_lst, prompt_lst

def llm_loop_gpt4(
        prompt_template,
        human_template,
        lst,
        prompt_condense,
        settings=LLMSettings
    ):
    """Loops through list, calling llm_router_GPT4 on each item and sleeping for tokens."""
    output_list = []
    prompt_lst = []
    count = 1
    for item in lst:
        logger.debug("llm_loop_gpt4: Processing item %s of %s", count, len(lst))
        output, total_tokens, model, prompt = llm_router_gpt4(
            prompt_template,
            human_template,
            item,
            prompt_condense,
            settings=settings
        )
        output_list.append(output["text"])
        prompt_lst.extend(prompt)
        count += 1
        sleep_for_tokens(total_tokens, model)
    return output_list, prompt_lst
