import pandas as pd
import logging
from summarizer import Summarizer
from goal import GoalExplorer
from vizgen import VizGenerator
from evaluator import VizEvaluator
from executor import ChartExecutor
from typing import Union, Optional
from datamodel import (Summary, ChartExecutorResponse, Persona, Goal)
from openai_textgen import TextGenerator
from utils import read_dataframe

class Manager:

    def __init__(self, text_gen: TextGenerator = None, data: Union[pd.DataFrame, str] = None) -> None:
        
        self.text_gen = text_gen or TextGenerator()
        self.summarizer = Summarizer()
        self.goal = GoalExplorer()
        self.vizgen = VizGenerator()
        self.executor = ChartExecutor()
        self.evaluator = VizEvaluator()

        if isinstance(data, str):
            file_name = data.split("/")[-1]
            self.data = read_dataframe(data)
        else:
            self.data = data


    def summarize(
        self,
        #data: Union[pd.DataFrame, str],
        file_name: Optional[str] = "",
        n_samples: int = 3,
        summary_method: str = "default",
        textgen_config: dict = {"n":1, "temperature":0}
    ) -> Summary:
        """ Summarize data given a DataFrame or file path."""

        return self.summarizer.summarize(
            data=self.data, text_gen=self.text_gen, file_name=file_name, n_samples=n_samples, 
            summary_method=summary_method, textgen_config=textgen_config)
    

    def goals(
        self,
        summary: dict,
        textgen_config: dict,
        n: int =5, 
        persona: Persona = None
    ) -> list[Goal]:
        """ Generate goals based on a sumamry and persona"""
        
        if isinstance(persona, str):
            persona = Persona(person=persona, rationale="")
        if isinstance(persona, dict):
            persona = Persona(**persona)

        return self.goal.generate(
            summary=summary, textgen_config=textgen_config, text_gen=self.text_gen, n=n, persona=persona
        )


    # def execute(
    #     self,
    #     code_specs: List[str],
    #     data: Any,
    #     library="matplotlib",
    #     return_error: bool = False,
    # ) -> Any:
    #      return self.executor.execute()
         

    def visualize(
        self,
        summary: dict,
        goal: Goal,
        textgen_config: dict,
        library: str = 'matplotlib',
        return_error: bool = False
    ):
        if isinstance(goal, dict):
            goal = Goal(**goal)
        if isinstance(goal, str): 
            goal = Goal(question=goal, visualization=goal, rationale="")
        
        code_specs = self.vizgen.generate(
            summary=summary, 
            goal=goal, 
            textgen_config=textgen_config, 
            text_gen=self.text_gen, 
            library=library
        )

        return self.executor.execute(code_specs=code_specs, data=self.data, library=library, return_error=return_error)
    
    def evaluate(
        self, 
        code: str,
        goal: Goal,
        summary: dict,
        textgen_config: dict,
        library: str = 'matplotlib'      
    ):
        if isinstance(goal, dict):
            goal = Goal(**goal)
        if isinstance(goal, str): 
            goal = Goal(question=goal, visualization=goal, rationale="")
        
        return self.evaluator.generate(code=code, goal=goal, textgen_config=textgen_config, summary=summary, text_gen=self.text_gen, library=library)

    

if __name__ == "__main__":
    import manager as m
    llm_config = {"n":1, 'max_tokens':2000, "temperature": 0, }
    text_gen = TextGenerator()
    df = pd.read_excel("../data/ROBERT_KING.xlsx", index_col=0)
    nlviz = m.Manager(text_gen=text_gen, data=df)
    print("creating data summary...")
    data_summary = nlviz.summarize(df, textgen_config=llm_config, summary_method="llm")
    question = "what is the distribution of transaction purposes ?"
    print(f"data_summary: {data_summary}")
    print(f"question asked: {question}")
    print("Generating the plot ....")
    charts = nlviz.visualize(summary=data_summary, goal=question, textgen_config=llm_config, return_error=True)
