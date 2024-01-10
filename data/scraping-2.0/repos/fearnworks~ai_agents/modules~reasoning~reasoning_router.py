from langchain import PromptTemplate, LLMChain
from .re_act import ReActStrategy, get_re_act_config
from .tree_of_thought import TreeOfThoughtStrategy, get_tot_config
from .chain_of_thought import ChainOfThoughtStrategy, get_cot_confg
from .reasoning_strategy import ReasoningConfig
from typing import Tuple, Callable, Optional
import pprint
import re
import os

class ReasoningRouter:
    def __init__(self, api_key: str, config: ReasoningConfig, question:str, display: Callable):
        """
        Initializes a ReasoningRouter instance.

        Args:
            question (str): The user's question.

        Returns:
            None
        """
        print("Creating Reasoning Router with config: ",)
        pprint.pprint(vars(config))
        self.api_key = api_key
        self.llm  = config.llm_class(temperature=config.temperature, max_tokens=config.max_tokens)
        self.question: str = question
        self.display: Callable = display


        self.strategies = {
            1: ReActStrategy(get_re_act_config(temperature=config.temperature), display=self.display),
            2: TreeOfThoughtStrategy(get_tot_config(temperature=config.temperature),display=self.display),
            3: ChainOfThoughtStrategy(get_cot_confg(temperature=config.temperature),display=self.display)
        }
        self.usage_block = f"""

        1. {self.strategies[1].usage} [1].
        2. {self.strategies[2].usage} [2].
        3. {self.strategies[3].usage} [3].

        """
        self.template = """
        Consider the following problem or puzzle: {question}. Based on the characteristics of the problem,
        identify the most suitable approach among the three techniques described below. consider each carefully 
        in the context of the question, write out the likelihood of success of each, and then select the most 
        appropriate approach:""" + self.usage_block + """
            Based on the characteristics of the given problem or puzzle, select the technique that aligns most closely with the nature of the problem. It is important to first provide the number of the technique that best solves the problem, followed by a period. Then you may provide your reason why you have chosen this technique. 

        The number of the selected technique is...
        """
        
    @staticmethod
    def find_first_integer(text: str) -> Optional[int]:
        """Finds the first integer in a string.

        Args:
            text (str): The string to search for an integer.

        Returns:
            int or None: The first integer found in the string, or None if no integer is found.
        """
        match = re.search(r'\d+', text)
        if match:
            return int(match.group())
        else:
            return None

    def determine_and_execute(self) -> Tuple[str, str]:
        """
        Determines the appropriate reasoning strategy based on the user's question and executes it.

        Returns:
            response : Reason the strategy was selected
            strat_response : Response from the strategy 
        """
        
        prompt = PromptTemplate(template=self.template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)

        response = llm_chain.run(self.question)
        print(response)
        self.display(response)
        n = self.find_first_integer(response)

        if n in self.strategies:
            strat_resp = self.strategies[n].run(self.question)
        else:
            strat_resp = (f"Strategy number {n} is not recognized.")
            print(strat_resp)
            
        return response, strat_resp

def get_reasoning_router_config(temperature: float = 0.6) -> ReasoningConfig:
    usage="This router should be used when determing the most effective strategy for a query requiring more complex, but general reasoning to derive"
    return ReasoningConfig(temperature=temperature, max_tokens=3000, usage=usage)