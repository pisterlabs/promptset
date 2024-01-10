import sys
import json
import openai
from semantic_search import semsearch
import helper_funcs as helper
import doc_reader as reader

MAX_TOKENS = 3500

def valid_gpt_input(cand_json: str) -> bool:
    """
    Checks if the candidate json input fits in GPT token input of 3500 total tokens.
    Choice is arbitrary to allow for 597 total tokens for setting system prompts and 
    GPT output.

    Parameters
    -----------
        cand_json (str) : Candidate json input to test

    Returns
    -----------
        fits_in_gpt (bool) : Boolean flag on whether the candidate can be passed
                    into ChatGPT API
    """
    token_count = len(cand_json) / 4

    return token_count < MAX_TOKENS

def format_gpt_input(gpt_docs: dict) -> str:
    """
    Formats the documentation to be passed into ChatGPT API into a joined
    sentence structure to minimize GPT API token input.

    Parameters
    -----------
        gpt_docs (dict) : Formatted documentation to be inputted into ChatGPT API

    Returns
    -----------
        gpt_input (str) : json-formatted dictionary keyed by the same filenames,
                but the the content is now strings instead of lists of string tokens
    """
    return json.dumps({file : " ".join(content) for file, content in gpt_docs.items()})

def optimize_gpt_input(gpt_docs: dict) -> str:
    """
    Optimizes the documentation passed into ChatGPT by filling in the maximum
    number of tokens that can be fit from five files. If the five most relevant
    files go over the maximum number of GPT API tokens, this function cuts out
    the least relevant file, and tries again until the json-formatted dictionary
    can be passed into the ChatGPT API with as many relevant documents as possible
    and while staying below the maximum token count.

    Parameters
    -----------
        gpt_docs (dict) : Formatted documentation to be inputted into ChatGPT API

    Returns
    -----------
        gpt_input (str) : json-formatted string dictionary that has been trimmed
                from lest relevant to most in order to fit into ChatGPT API
                max token count of 3597 for inputs.
    """
    temp_docs = gpt_docs.copy()
    cand = format_gpt_input(temp_docs)

    while not valid_gpt_input(cand):
        file_keys = list(temp_docs.keys())
        file_to_remove = file_keys[-1]
        temp_docs.pop(file_to_remove)
        cand = format_gpt_input(temp_docs)

    return cand


def replace_filenames_with_links(gpt_output: str, hyperlinks: dict) -> str:
    """
    Replaces filenames in Ask PW output with hyperlinks in markdown format.

    Parameters
    -----------
        gpt_output (str) : output from ChatGPT PW API
        hyperlinks (dict) : the hyperlinks corresponding to the file names

    Returns
    -----------
        formatted_output (str) : gpt_output but with file names replaced with hyperlinks
    """
    formatted_output = gpt_output

    for filename, link in hyperlinks.items():
        hyperlink = f"[{filename}]({link})"
        formatted_output = formatted_output.replace(filename, hyperlink)

    return formatted_output

def run_gpt(query, formatte_docs, api_key=helper.get_api_key()):
    """
    Function that runs the gpt-3.5-turbo AI API on a query and set of arguments
    Arguments should consist of a variable length list, where each
    element contains a list of tokens from the most relevant files related to
    the inputted query.

    Paramaters:
        query (str) : inputted query from user
        formatted_docs (list[str]) : json-formatted dictionary containing file and
                content info from semantic search
        api_key (str) : user API key to run
    
    Returns:
        reply (str) : GPT AI response to query with supporting relevant documents
    """
    openai.api_key = api_key

    gpt_prompt = "You are a helpful assistant in charge of helping users understand our platform."
    clarification_1 = "Your responses should not require users to search through our files and should be fully comprehensive. However, always include the most relevant files to the query at the bottom if the user would like to conduct further reading."
    clarification_2 = "If the inputted query isn't related to PW documentation, respond explaining that you are meant as an assistant for the Parallel Works platform. Tangentially related queries are okay."
    clarification_3 = "If the message passed to you is `Your query does not match anything in our system.`, explain that we currently don't have documentation related to that query."

    messages = [
        {"role": "system", "content": gpt_prompt},
        {"role": "system", "content": clarification_1},
        {"role": "system", "content": clarification_2},
        {"role": "system", "content": clarification_3},
        {"role": "user", "content": query}
    ]

    messages.append({"role": "user", "content": formatte_docs})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    reply = response.choices[0].message.content
    return reply

def main():
    if len(sys.argv) < 2:
        raise ValueError("Query input required.")

    query = sys.argv[1]

    docs_path = sys.argv[2] if len(sys.argv) >= 3 else "../data/docs"
    preproc_docs = helper.read_clean_process_data(docs_path)
    hyperlink_dict = reader.create_hyperlink_dict(docs_path)

    w2v_model = helper.load_w2v()
    # vectorizer, tfidf_matrix = helper.load_tfidf()

    # ss_docs = semsearch(query, preproc_docs, w2v_model, vectorizer, tfidf_matrix)
    ss_docs = semsearch(query, preproc_docs, w2v_model)

    gpt_input = optimize_gpt_input(ss_docs)

    reply = run_gpt(query, gpt_input)

    hyperlink_reply = replace_filenames_with_links(reply, hyperlink_dict)

    print(f"{hyperlink_reply}\n")

if __name__ == "__main__":
    main()

