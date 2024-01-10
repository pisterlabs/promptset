from .reasoning_strategy import ReasoningStrategy
from langchain import LLMChain, PromptTemplate
from .reasoning_strategy import ReasoningStrategy, ReasoningConfig
from typing import Callable
import pprint

class TreeOfThoughtStrategy(ReasoningStrategy):
    def __init__(self, config: ReasoningConfig, display: Callable):
        super().__init__(config=config, display=display)
        print("Creating Reasoning Router with config: ",)
        pprint.pprint(vars(config))

    def run(self, question)-> str:
        print('Using ToT')
        self.display("Using 'Tree of Thoughts'")

        template_tot = """Imagine three different experts are answering this question.
    They will brainstorm the answer step by step reasoning carefully and taking all facts into consideration
    All experts will write down 1 step of their thinking,
    then share it with the group.
    They will each critique their response, and the all the responses of others
    They will check their answer based on science and the laws of physics
    Then all experts will go on to the next step and write down this step of their thinking.
    They will keep going through steps until they reach their conclusion taking into account the thoughts of the other experts
    If at any time they realise that there is a flaw in their logic they will backtrack to where that flaw occurred 
    If any expert realises they're wrong at any point then they acknowledges this and start another train of thought
    Each expert will assign a likelihood of their current assertion being correct
    Continue until the experts agree on the single most likely answer and write out that answer along with any commentary to support that answer
    The question is {question}

    The experts reasoning, along with their final answer is...
    """
        prompt = PromptTemplate(template=template_tot, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        response_tot = llm_chain.run(question)
        print(response_tot)
        self.display(response_tot)
        return response_tot

def get_tot_config(temperature: float = 0.7) -> ReasoningConfig:
    usage= """
    This problem is complex and the solution requires exploring multiple reasoning paths over thoughts.
      It treats the problem as a search over a tree structure, with each node representing a partial 
      solution and the branches corresponding to operators that modify the solution. It involves thought 
      decomposition, thought generation, state evaluation, and a search algorithm 
    """
    return ReasoningConfig(usage=usage, temperature=temperature)