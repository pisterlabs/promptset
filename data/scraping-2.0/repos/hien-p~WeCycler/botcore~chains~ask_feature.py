


ASK_FEATURE_CONST = \
{"inputs":["product", "n_top"],
 "outputs": {"questions": """a js array of questions."""},
"template": """You are interesting in a {product}.
Please ask top {n_top} questions about the features of the {product}.
{format_instructions}
Questions:"""}

from langchain.llms import BaseLLM
from langchain import LLMChain
import sys
import os
sys.path.append(f"{os.path.dirname(__file__)}/../..")
from botcore.utils.prompt_utils import build_prompt

def build_ask_feature_chain(model: BaseLLM):
    inputs = ASK_FEATURE_CONST['inputs']
    outputs = ASK_FEATURE_CONST['outputs']
    template = ASK_FEATURE_CONST['template']
    prompt = build_prompt(inputs, outputs, template, include_parser=True)
    chain = LLMChain(llm=model, prompt=prompt, output_key='result')
    return chain
