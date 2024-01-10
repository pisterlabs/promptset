import time
import os
import json
import logging as log
from openai import OpenAI


def gpt_cost_calculator(
    inp_tokens: int, out_tokens: int, model: str = "gpt-3.5-turbo"
) -> int:
    """
    Calculate the cost of the GPT API call
    """
    cost = 0
    # GPT-3.5 Turbo
    if model == "gpt-3.5-turbo":
        input_cost = 0.0010
        output_cost = 0.0020
        cost = ((inp_tokens * input_cost) + (out_tokens * output_cost)) / 1000
    # GPT-4
    elif model == "gpt-4":
        input_cost = 0.03
        output_cost = 0.06
        cost = ((inp_tokens * input_cost) + (out_tokens * output_cost)) / 1000
    else:
        log.error("Invalid model")

    return cost


def sanitize_search_query(prompt: str,open_api_key:str, location: str = None) -> json:
    """
    Sanitize the search query using OpenAI for web search
    """
    t_flag1 = time.time()

    if open_api_key is None:
        try:
            open_api_key = os.getenv("OPENAI_API_KEY")
        except Exception as e:
            log.error(f"No Open API key found")
            raise e

    prompt = f"{prompt.strip()}"
    client = OpenAI(api_key=open_api_key)
    # system_prompt = "Convert the user goal into a useful web search query, for finding the best contacts for achieving the Goal through a targeted web search, include location if needed. The output should be in JSON format, also saying where to search in a list, an enum (web, yelp), where web is used for all cases and yelp is used only for restaurants, home services, auto service, and other services and repairs."
    system_prompt = """Understand the goal and provide the best solution task, and a web search query for the solution for solving and helping the user achieve their goals. Include the location if needed. 
The solution should be based on the finding the best person or service to contact for helping or completing the user goal. 
The output should be in JSON format, also saying where to search in a list, an enum (web, yelp, gmaps), where web is used for all cases and yelp is used only for restaurants, home services, food, and other services and repairs. `gmaps` is Google Maps, who can retrieve info about businesses and services in a location."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {
                    "role": "user",
                    "content": "Location: Kochi, Kerala;\nGoal: I want a good chef for my anniversary party for 50 people.",
                },
                {
                    "role": "system",
                    "content": '{"solution":"Search for all event chefs in Kochi Kerala, to email and call them", "search_query":"Event Chefs in Kochi, Kerala", "search":["web", "yelp"]}',
                },
                {
                    "role": "user",
                    "content": "Location: Oakland, CA;\nGoal: I want a SUV car for rent for 2 days, for a trip to Yosemite.",
                },
                {
                    "role": "system",
                    "content": '{"solution":"Search for all Car rental service in Oakland, CA, Who can give SUV and find their contacts", "search_query":"SUVs car rental in Oakland, CA", "search":["web", "gmaps"]}',
                },
                {
                    "role": "user",
                    "content": "Location: - ;\nGoal: I need an internship in UC Davis in molecular biology this summer.",
                },
                {
                    "role": "system",
                    "content": '{"solution": "Search for all UC Davis professors specializing in molecular biology research, to email them", "search_query":"Professors UC Davis molecular biology and internship contacts.", "search":["web"]}',
                },
                {"role": "user", "content": f"Location: {location};\nGoal: {prompt}"},
            ],
        )
    except Exception as e:
        log.error(f"Error in OpenAI query sanitation: {e}")
        exit(1)

    t_flag2 = time.time()
    log.info(f"OpenAI sanitation time: {t_flag2 - t_flag1}")

    # tokens used
    tokens_used = response.usage.total_tokens
    cost = gpt_cost_calculator(
        response.usage.prompt_tokens, response.usage.completion_tokens
    )
    log.info(f"Tokens used: {tokens_used}")
    log.info(f"Cost for search query sanitation: ${cost}")
    try:
        result = json.loads(response.choices[0].message.content)
    except Exception as e:
        log.error(f"Error parsing json: {e}")
        result = {}
    return result