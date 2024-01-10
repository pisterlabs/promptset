CHAT_ROUTE_TEMPLATE = """Given a raw text input to a language model select the model prompt best suited for the input. You will be given the names of the available prompts and a description of what the prompt is best suited for. You may also revise the original input if you think that revising it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \ name of the prompt to use or "DEFAULT"
    "next_inputs": string \ a potentially modified version of the original input
    "question": string \ the original input
}}}}
```

REMEMBER: "destination" MUST be one of the candidate prompt names specified below OR it can be "DEFAULT" if the input is not well suited for any of the candidate prompts.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT >>"""

import sys
import os
from langchain.chains.router.llm_router import LLMRouterChain
from langchain.prompts import PromptTemplate
from langchain.chains.router import MultiPromptChain
sys.path.append(f"{os.path.dirname(__file__)}/../..")
from botcore.chains.assess_usage import build_assess_elec_usage
from botcore.chains.pros_cons import build_pros_cons_chain
from botcore.utils.memory_utils import QAMemory
from botcore.routing.chat_route_parser import ChatRouterOutputParser 

class ProductChatRouter():

    def __init__(self, model, qa_memory: QAMemory):
        self.bot_memory = qa_memory
        self.model = model
        self.assess_usage = build_assess_elec_usage(model, self.bot_memory.memory)
        self.pros_cons = build_pros_cons_chain(model, self.bot_memory.memory)
        print("Router ready")

    def get_const(self):
        prompt_infos = [
            {
                "name": "assess electronic usage",
                "description": "Good for answering questions about electronic product usage.",
                "chain": self.assess_usage,
            },
            {
                "name": "pros and cons",
                "description": "Good for answering questions about the pros and cons of a product.",
                "chain": self.pros_cons,
            },
        ]
        return prompt_infos

    def build_destinations(self):
        
        prompt_infos = self.get_const()
        destination_chains = {}

        for p_info in prompt_infos:
            name = p_info["name"]
            chain = p_info['chain']
            destination_chains[name] = chain
        
        default_chain = self.pros_cons
        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        return destinations_str, destination_chains, default_chain

    def build_router(self):
        dest_str, dest_chains, default_chain = self.build_destinations()
        router_template = CHAT_ROUTE_TEMPLATE.format(destinations=dest_str)
        router_prompt = PromptTemplate(template=router_template, input_variables=["input"], output_parser=ChatRouterOutputParser())
        router_chain = LLMRouterChain.from_llm(self.model, router_prompt)

        self.chain = MultiPromptChain(router_chain=router_chain, destination_chains=dest_chains,
                                 default_chain=default_chain, verbose=True) 
        print("Build done")
