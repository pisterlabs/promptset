from typing import Dict, List
from collections.abc import Callable

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from chains import llm, model

class StanceChain(Chain):
    generation : LLMChain
    model : Callable[[str], str]
    prompt : PromptTemplate
    
    @property
    def input_keys(self) -> List[str]:
        return ['stances']

    @property
    def output_keys(self) -> List[str]:
        return ['result', 'prompt']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        if "stances" in inputs:
            inputs["generation"] = self.generation.run(stances=inputs["stances"])
        t5_prompt = self.prompt.format(**{k: v for k,v in inputs.items() if k in self.prompt.input_variables})
        t5_result = self.model(t5_prompt)
        return {'result': t5_result, 'prompt': t5_prompt}


summarizationPrompt = PromptTemplate(
    input_variables=["stances"],
    template="""Write a general comment that expresses the following stances: {stances}""",
)


openAIChain = LLMChain(llm=llm, prompt=summarizationPrompt)

stancedetectiontemplate = PromptTemplate(
    input_variables=["generation"],
    template="""Parent summary: What do you think?\n A low toxicity post: That is just my opinion. What do you think?\nReply summary: {generation}\nA low toxicity reply:""",
)

stance_detection_chain = StanceChain(generation=openAIChain, prompt=stancedetectiontemplate, model=model)

stance_detection = lambda x: {k: v for k, v in stance_detection_chain(x).items() if k in ["result", "prompt"]}