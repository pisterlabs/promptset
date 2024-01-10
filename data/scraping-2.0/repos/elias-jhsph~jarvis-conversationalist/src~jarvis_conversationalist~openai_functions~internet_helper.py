import json
import multiprocessing
import threading
import requests
from bs4 import BeautifulSoup
from openai import OpenAI, _utils
from googlesearch import search as google_search
from tiktoken import encoding_for_model

client = OpenAI()
_utils._logs.logger.setLevel("CRITICAL")

basic_model = "gpt-3.5-turbo-16k"
advanced_model = "gpt-4"
enc = encoding_for_model(advanced_model)
temperature = 0.6


def search(search_term: str, num_results: int = 10, advanced: bool = False) -> dict:
    """
    Searches for a term using either Google Custom Search API or a free alternative.

    :param search_term: The term to search for.
    :type search_term: str
    :param num_results: The number of results to return, defaults to 10.
    :type num_results: int, optional
    :param advanced: Whether to use advanced info, defaults to False.
    :type advanced: bool, optional
    :return: A dictionary containing the search results.
    :rtype: dict
    """
    search_results = []
    for url in google_search(search_term, num_results=num_results, advanced=advanced):
        if advanced:
            search_results.append({"link": url.url, "title": url.title, "description": url.description})
        else:
            search_results.append({"link": url})
    return {"items": search_results}


def refine_query(query: str) -> str:
    """
    Refines a query using the OpenAI API.

    This function is used to refine a query to get better search results. It uses the OpenAI API to ask the user
    for context and keywords to add to the query. The user's response is then sent directly to google.

    :param query: The query to refine.
    :type query: str
    :return: The refined query.
    :rtype: str
    """
    response = client.chat.completions.create(model=advanced_model,
    messages=[{"role": "user", "content":
               f"Please help me improve this search query for better results: '{query}'. Add context and keywords "
               "you think help better capture the idea behind the query. The response you send will go directly "
               "into google. Here is a helpful reminder of google tools you can use but consider not using them "
               "if you don't think you need them. Make sure some keywords aren't in quotes or you risk "
               "only getting results with those exact words in that order:\n\n"
               'Quotes (""): Use quotes to search for an exact phrase or word order.\n'
               "Minus (-): Exclude a specific word from your search.\n"
               "Asterisk (*): Use as a placeholder for unknown words.\n"
               "OR: Search for multiple terms or phrases.\n"
               "intitle: (intitle:): Search for words specifically in the title of webpages.\n"
               "intext: (intext:): Search for words specifically in the body of webpages.\n"
               "Note: Do not be so specific in your search that you miss the general point of the query. Also "
               "DO NOT SURROUND THE ENTIRE QUERY BY QUOTES.\n Query:"}],
    max_tokens=100,
    n=1,
    temperature=temperature)
    refined_query = response.choices[0].message.content
    return refined_query


def extract_content(url: str) -> str:
    """
    Extracts the content from a webpage using BeautifulSoup.

    :param url: The URL of the webpage.
    :type url: str
    :return: The content of the webpage.
    :rtype: str
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])
        return content
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""


def summarize(content: str, refined_query: str) -> str:
    """
    Summarizes a piece of text using the OpenAI API.

    :param content: The text to summarize.
    :type content: str
    :param refined_query: The refined query.
    :type refined_query: str
    :return: The summary.
    :rtype: str
    """
    response = client.chat.completions.create(model=basic_model,
    messages=[{'role': 'system', 'content': f'There was a search for the following query:\n"{refined_query}"\nPlease '
                                          f'provide a concise summary of the following content while keeping mind '
                                          f'what will best respond to the search query:\n{content}\n'}],
    max_tokens=400,
    n=1,
    stop=None,
    temperature=temperature)
    summary = response.choices[0].message.content
    return summary


def rank_relevance(url: str, summary: str, query: str) -> int:
    """
    Ranks the relevance of a summary using the OpenAI API.

    :param url: The URL of the webpage.
    :type url: str
    :param summary: The summary.
    :type summary: str
    :param query: The query.
    :type query: str
    :return: The relevance of the summary.
    """
    prompt = f"Given the query '{query}', rate the relevance of this summary from 1 (not relevant) to 10 (highly " \
             f"relevant):\nURL: {url}\nSummary: {summary}"

    schema = {"type": "function",
              "function": {
                    "name": "store_rank_relevance",
                    "description": "Stores the relevance of a summary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "relevance": {
                                "type": "number",
                                "description": "The relevance of the summary. relevance is a number from 1 to 10.",
                            },
                        },
                        "required": ["relevance"],
                    },
              }
              }

    response = client.chat.completions.create(model=advanced_model,
    messages=[{'role': 'system', 'content': prompt}],
    max_tokens=100,
    n=1,
    stop=None,
    temperature=temperature,
    tools=[schema],
    tool_choice={"type": "function", "function": {"name": "store_rank_relevance"}})
    relevance_raw = response.choices[0].message.tool_calls[0].function.arguments
    relevance = int(json.loads(relevance_raw)['relevance'])
    return relevance


def synthesize_information(summaries: list, query: str) -> str:
    """
    Synthesizes information from a list of summaries using the OpenAI API.

    :param summaries: The list of summaries.
    :type summaries: list
    :param query: The query.
    :type query: str
    """
    summaries_text = "\n".join([f"Summary {i + 1}: {summary}" for i, (url, summary) in enumerate(summaries)])
    response = client.chat.completions.create(model=advanced_model,
    messages=[{"role": "system", "content": f"Given the following summaries about '{query}', please synthesize "
                                            f"a coherent and comprehensive response:\n{summaries_text}\n"}],
    max_tokens=500,
    n=1,
    temperature=temperature)
    synthesized_info = response.choices[0].message.content
    return synthesized_info


def truncate_content(content: str, max_tokens: int = 3500) -> str:
    """
    Truncates a piece of text to a maximum number of tokens.

    :param content: The text to truncate.
    :type content: str
    :param max_tokens: The maximum number of tokens.
    :type max_tokens: int
    :return: The truncated text.
    :rtype: str
    """
    tokens = enc.encode(content)

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        truncated_content = enc.decode(tokens)
        return truncated_content + "(TRUNCATED)"
    else:
        return content


def search_helper(query: str, result_number: int = 6, skip: threading.Event = None) -> dict:
    """
    Helper function for search.

    This function is used to run the search for a given query by first refining the query, then searching for the query,
    then summarizing the results, then ranking the relevance of the summaries, and finally synthesizing the information

    :param query: The query.
    :type query: str
    :param result_number: The number of results to return.
    :type result_number: int
    :param skip: A threading.Event object that can be used to stop the search.
    :type skip: threading.Event
    :return: A dictionary containing the search results.
    :rtype: dict
    """
    search_data = {"initial_query": query, "refined_query": refine_query(query), "search_results": [],
                   "ranked_summaries": [], "synthesized_information": None}

    temp = search(query, num_results=result_number)
    if "items" not in temp:
        search_data["refined_query"] = query
        temp = search(search_data["refined_query"], num_results=result_number)
    search_data["search_results"] = temp['items']

    for result in search_data["search_results"]:
        if skip is not None:
            if skip.is_set():
                return {}
        content = extract_content(result['link'])
        summary = summarize(truncate_content(content), search_data["refined_query"])
        snippet = result.get('snippet', '')  # Use an empty string if snippet is not available
        search_data["ranked_summaries"].append({"url": result['link'], "content": content,
                                                "summary": summary, "snippet": snippet})

    for summary_data in search_data["ranked_summaries"]:
        if skip is not None:
            if skip.is_set():
                return {}
        relevance = rank_relevance(summary_data["url"], summary_data["summary"], search_data["refined_query"])
        summary_data["relevance"] = relevance

    search_data["ranked_summaries"].sort(key=lambda x: x["relevance"], reverse=True)

    if skip is not None:
        if skip.is_set():
            return {}

    search_data["synthesized_information"] = synthesize_information(
        [(data["url"], data["summary"]) for data in search_data["ranked_summaries"]],
        search_data["refined_query"]
    )

    return search_data


def simplify_output(search_data: dict) -> dict:
    """
    Simplifies the output of the search function.

    :param search_data: The output of the search function.
    :type search_data: dict
    :return: A simplified version of the output of the search function.
    :rtype: dict
    """
    simplified_output = {k: v for k, v in search_data.items() if k != "summaries"}
    for summary_data in simplified_output["ranked_summaries"]:
        summary_data.pop("content", None)
    return simplified_output


def generate_final_prompt(simplified_output: dict, max_tokens: int = 1800) -> str:
    """
    Generates the final prompt for the chatbot.

    This function is used to generate the final prompt for the chatbot by combining the information from the search
    function.

    :param simplified_output: The simplified output of the search function.
    :type simplified_output: dict
    :param max_tokens: The maximum number of tokens.
    :type max_tokens: int
    :return: The final prompt for the chatbot.
    :rtype: str
    """
    synthesized_information = simplified_output["synthesized_information"]
    ranked_summaries = simplified_output["ranked_summaries"]
    refined_query = simplified_output["refined_query"]
    user_query = simplified_output["initial_query"]

    ranked_summaries_text = "\n".join(
        [f"{i + 1}. {summary['url']} (Relevance: {summary['relevance']}):\n{summary['summary']}"
         for i, summary in enumerate(ranked_summaries)]
    )

    pre_prompt = (
        f"The user has requested a response to the following query '{user_query}'.\n"
        f"An AI language model working with you has conducted an internet search for '{refined_query}', "
        f"which was based on the provided user query. "
        f"It has synthesized the following information from the search results: '{synthesized_information}'. "
        f"Here are the ranked summaries of the top search results:\n\n"
    )

    post_prompt = (
        f"\n\n"
        f"Please analyze these results and provide the most appropriate response to the User.\n"
        f"Consider the following options:\n"
        f"1. Pass along the final summary\n"
        f"2. Provide a very short final answer\n"
        f"3. Suggest specific websites for further reading\n"
        f"4. Recommend a deeper search or further inquiry\n"
        f"5. Offer color commentary on the findings\n"
        f"6. Combine any of the above options.\n"
        f"NOTE: Provide the exact response that you would have me give the user. DO NOT mention which approach you "
        f"have chosen. Give your response exactly as you would give it to the end user.\n\n"
        f"Remember the user doesn't have access to the results above, so any text you want to refer to from above "
        f"you must reiterate that information to the user in your own words! "
        f"And don't forget your first system message (NO FULL URLS)! Good luck!"
    )

    prompt = pre_prompt + ranked_summaries_text + post_prompt

    tokens = enc.encode(prompt)
    if len(tokens) > max_tokens:
        diff = len(tokens) - max_tokens
        new = enc.encode(ranked_summaries_text)
        if len(new) < diff+10:
            raise Exception("Could not shrink internet final prompt within limit!")
        prompt = pre_prompt + truncate_content(ranked_summaries_text, len(new) - (diff+10)) + post_prompt
    return prompt


def create_internet_context(query: str, result_number: int = 10,
                            max_tokens: int = 1800, skip: threading.Event = None) -> tuple:
    """
    Creates the internet context for the chatbot.

    This function is used to create the internet context for the chatbot by combining the information from the search
    function. Then it generates the final prompt for the chatbot. Then it returns the final prompt and the simplified
    output of the search function.

    :param query: The query to search for.
    :type query: str
    :param result_number: The number of results to return.
    :type result_number: int
    :param max_tokens: The maximum number of tokens.
    :type max_tokens: int
    :param skip: The skip object.
    :type skip: Skip
    :return: The final prompt for the chatbot and the simplified output of the search function.
    :rtype: tuple
    """
    if skip is None:
        skip = multiprocessing.Event()
    if skip.is_set():
        return "Sorry."
    if skip is not None:
        if skip.is_set():
            return "", {}
    search_data = search_helper(query, result_number=result_number, skip=skip)
    if skip is not None:
        if skip.is_set():
            return "", {}
    simplified_output = simplify_output(search_data)
    if skip is not None:
        if skip.is_set():
            return "", {}
    result = generate_final_prompt(simplified_output, max_tokens=max_tokens)
    return result, simplified_output
