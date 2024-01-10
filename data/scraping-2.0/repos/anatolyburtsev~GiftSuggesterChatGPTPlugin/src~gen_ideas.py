import json
import logging
import re

import dirtyjson
from langchain import PromptTemplate
from langchain.chains import (
    LLMChain,
    SimpleSequentialChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
from langchain.utilities import GoogleSearchAPIWrapper

logger = logging.getLogger(__name__)


def generate_ideas(llm: ChatOpenAI, query: str) -> list[str]:
    system_message_1 = """You are an owner of the biggest gift shop in the world. You know your business inside out and 
    can find a perfect gift for anybody. You are going to be asked to provide 3 best gift ideas for the given request.
    """
    template1 = "Please help me find a gift for the following request: '{request}'"

    prompt_template1 = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_message_1),
            HumanMessagePromptTemplate.from_template(template1),
        ]
    )
    get_ideas_chain = LLMChain(llm=llm, prompt=prompt_template1)

    system_message_2 = """You are experienced developer. You got a text with some gift ideas, you need to transform it to json 
    format like that:
    ideas: ["idea1", "idea2", "idea3"]

    Example:
    Input:
    Based on the information provided, here are three gift ideas:\n\n1. Fitness tracker: A fitness tracker is a great gift for someone who enjoys walking. It can help your spouse keep track of their steps, distance, and calories burned.\n\n2. Walking shoes: A good pair of walking shoes is essential for anyone who enjoys walking. You can choose a comfortable and stylish pair that your spouse will love.\n\n3. Personalized water bottle: Staying hydrated is important when walking, and a personalized water bottle is a great way to make sure your spouse always has water on hand. You can customize it with your spouse's name or a special message.\n\nDo you need more information or are there any specific details I should know about?
    Output:
    ideas: ["fitness tracker", "Walking shoes", "Personalized water bottle"]

    Example:
    Input:
    How about a new pair of comfortable walking shoes or a Fitbit to track her steps and progress? Another idea could be a subscription to a hiking or walking trail guidebook or a membership to a local nature park or trail.
    Output:
    ideas: ["walking shoes", "Fitbit", "hiking or walking trail guidebook", "membership to a local nature park or trail"]
    """
    template2 = "Transform to json the following statement with ideas: {statement}"

    prompt_template2 = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_message_2),
            HumanMessagePromptTemplate.from_template(template2),
        ]
    )

    transform_json_chain = LLMChain(llm=llm, prompt=prompt_template2)

    get_ideas_json_chain = SimpleSequentialChain(
        chains=[get_ideas_chain, transform_json_chain], verbose=True
    )

    list_of_ideas_raw = get_ideas_json_chain.run(query)

    try:
        ideas_list = json.loads(list_of_ideas_raw)["ideas"]
    except:
        pattern = r"(\[.*?\])"
        result = re.search(pattern, list_of_ideas_raw)
        ideas_list_string = result.group(1)
        logger.info(f"list prepared for json parsing: {ideas_list_string}")
        ideas_list = json.loads(ideas_list_string)
        logger.info(f"parsed list: {ideas_list}")

    return ideas_list


def clean_json_string(input_string):
    # Replace invalid escape sequences with valid ones
    input_string = re.sub(r"\\x", r"\\u00", input_string)
    return input_string


def get_links(llm: ChatOpenAI, search: GoogleSearchAPIWrapper, idea: str) -> list:

    search_results_all = search.results(idea, num_results=10)
    search_results = [x for x in search_results_all if "/dp/" in x["link"]]

    search_results_only_asin_links = [
        {
            # drop 'snippet' here because it's failing json parser sometime
            "link": x["link"],
            "title": x["title"],
        }
        for x in search_results
    ]
    logger.info(f"search_results_only_asin_links: {search_results_only_asin_links}")
    if not search_results_only_asin_links:
        return []

    search_results_only_titles = [x["title"] for x in search_results_only_asin_links]

    # Chain 4 choose best
    system_message_4 = """You are an experienced amazon shopper and know people very well. You would need to choose 2 
        best gifts from the list of ideas for a given request. Response should be array with element number.
        Example:
        ideas:  ['World of Tanks Halloween Tank Skull T-Shirt ... - Amazon.com', 'World of Tanks T57 Heavy Tank "Tried n\' True" T-Shirt ... - Amazon.com', 'Amazon.com: World of Tanks Skoda T 27 "Bohemian Warrior" T-Shirt', 'World of Tanks Santa & Tankdeer T-Shirt : Clothing ... - Amazon.com', 'World of Tanks CS-52 LIS Fox T-Shirt', 'World of Tanks Blitz Legendary Sherman T-Shirt', 'World of Tanks M48A5 Patton T-Shirt : Clothing ... - Amazon.com', 'World of Tanks Blitz Classy T-Shirt : Clothing, Shoes ... - Amazon.com']
        request: "gift for a friend who loves world of tanks game"
        response: [0, 3, 5]

        """
    template_4 = """
        ideas: {search_results}. 
        request: {request}. 
        response:  
        """

    prompt_template_4 = PromptTemplate(
        template=template_4, input_variables=["search_results", "request"]
    )
    prompt_4 = prompt_template_4.format(
        search_results=str(search_results_only_titles), request=idea
    )

    best_ideas_ids = llm(
        [SystemMessage(content=system_message_4), HumanMessage(content=prompt_4)]
    )
    cleaned_best_ideas_string = clean_json_string(best_ideas_ids.content)
    try:
        best_ideas_ids_list = dirtyjson.loads(cleaned_best_ideas_string)
    except dirtyjson.error.Error:
        logger.error(f"dirtyjson failed to parse string: {cleaned_best_ideas_string}")
        return []
    logger.info(f"parsed best ideas: {best_ideas_ids_list}")
    if not best_ideas_ids_list:
        return []

    best_ideas_list = [search_results[i] for i in best_ideas_ids_list][:3]

    if type(best_ideas_list[0]) == str or "link" not in best_ideas_list[0].keys():
        logger.error(f"GPT provided shrank list from google response: {best_ideas_list}")
        return []

    keys_to_keep = ["link", "title", "snippet"]
    best_ideas_result = [
        {k: best_idea_details[k] for k in keys_to_keep if k in best_idea_details}
        for best_idea_details in best_ideas_list
    ]
    logging.info(best_ideas_result)
    return best_ideas_result
