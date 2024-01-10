
import warnings
warnings.filterwarnings('ignore')

from langchain.chains import LLMChain, ConversationChain
from langchain.chains.router import MultiPromptChain
from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

from logger_setup import log

# Router Chain
high_priority_template = """ You are a very smart assistant. \
    You are great at creating items to add to monday.com with high priority. \

    Generate a dictionary of the following:
    Example: add buy pencils for the office to monday.com
    Output: title: name of the action, description: a short description with details max 20 words, priority: High

    {input}
    """

low_priority_template = """ You are a very smart assistant. \
    You are great at creating items to add to monday.com with low priority. \

    Generate a dictionary of the following:
    Example: add buy peanut butter to monday.com
    Output: title: name of the action, description: a short description with details max 20 words, priority: Low

    {input}
    """

prompt_infos = [
    {
        "name": "high_priority",
        "description": "Good for generating items with high priority when is work related",
        "prompt_template": high_priority_template,
    },
    {
        "name": "low_priority",
        "description": "Good for generating items with low priority when is personal related",
        "prompt_template": low_priority_template,
    },
]

def build_priority_multi_router_chain(LLM):

    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
        chain = LLMChain(llm=LLM, prompt=prompt)
        destination_chains[name] = chain

    log.info("Destination chains created")
    default_chain = ConversationChain(llm=LLM, output_key="text")

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(LLM, router_prompt)
    log.info("Router chain created")

    multi_router_chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )

    return multi_router_chain