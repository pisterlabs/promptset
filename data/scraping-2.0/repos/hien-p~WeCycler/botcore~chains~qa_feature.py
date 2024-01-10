ASK_FEATURE_CONST = \
{"inputs":["product", "n_top"],
        "outputs": {"chain": "always return 'ask_feature'","questions": """a js array of elements. Each element should contains 2 properties:
 question: str // the question.
 options: str // a js array of options for the question along with its correct unit. There should not be more than 5 options."""},
"template": """You are interesting in a {product}.
Please ask top {n_top} questions about the features of the {product}.
{format_instructions}
Questions:"""}

from langchain.llms import BaseLLM
from langchain import LLMChain
import sys
import os
sys.path.append(f'{os.path.dirname(__file__)}/../..')
from botcore.utils.prompt_utils import build_prompt

def build_ask_feature_chain(model: BaseLLM):
    """
    Chain designed for asking feature of a product

    Input: chain({"product": "rice cooker", "n_top": 5})
    
    """
    inputs = ASK_FEATURE_CONST['inputs']
    outputs = ASK_FEATURE_CONST['outputs']
    template = ASK_FEATURE_CONST['template']
    prompt = build_prompt(inputs, outputs, template, include_parser=False)
    chain = LLMChain(llm=model, prompt=prompt, output_key='result')
    return chain
