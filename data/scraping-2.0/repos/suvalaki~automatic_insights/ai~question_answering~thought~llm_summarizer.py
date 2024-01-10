from typing import Any, Dict, List, Optional
from itertools import chain

from pydantic import Extra
from pydantic_computed import Computed, computed

from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from pydantic import BaseModel, Field
from langchain import PromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from ai.question_answering.thought.base import (
    Thought,
    ThoughtPairCombination,
    ThoughtPairComparer,
    ThoughtSummarizer,
)


class LLMManyThoughtConclusion(BaseModel):
    conclusion: str


MANY_THOUGHT_SUMMARY_PROMPT_TEMPLATE = """
You are a data analyst tasked with summarizing thoughts into a coherent whole. 
You are provided non-contradictory thoughts. 
You are to summarize all of the important and distinct aspects of the thoughts.

The input thoughts follow:
{thoughts}


You are respond with your colusion based on the thoughts.

{format_instructions}
"""

MANY_THOUGHT_CONCLUSION_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=LLMManyThoughtConclusion
)

MANY_THOUGHT_SUMMARY_PROMPT = PromptTemplate(
    template=MANY_THOUGHT_SUMMARY_PROMPT_TEMPLATE,
    input_variables=["thoughts"],
    partial_variables={
        "format_instructions": MANY_THOUGHT_CONCLUSION_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=MANY_THOUGHT_CONCLUSION_OUTPUT_PARSER,
)


class LLMManyThoughtConclusionScore(BaseModel):
    score: float


MANY_THOUGHT_CONCLUSION_SCORING_PROMPT_TEMPLATE = """
You are a data analyst tasked with summarizing thoughts into a coherent whole. 
You are provided non-contradictory thoughts. 
You will be provided with a conclusion based on those thoughts. 
You are to score the colcusion based on the evidence supplied by the input thoughts.

The input thoughts follow:
{thoughts}

The conclusion summarized from the input thoughts follow:
{conclusion}


Answer only a score between 0 and 1. 
0.0 means the conclusion is completely wrong.
1.0 means the conclusion is completely correct.
0.5 means the conclusion is half correct.
< 0.5 means the conclusion is more wrong than right.
> 0.5 means the conclusion is more right than wrong.

{format_instructions}

"""

MANY_THOUGHT_CONCLUSION_SCORING_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=LLMManyThoughtConclusionScore
)

MANY_THOUGHT_CONCLUSION_SCORING_PROMPT = PromptTemplate(
    template=MANY_THOUGHT_CONCLUSION_SCORING_PROMPT_TEMPLATE,
    input_variables=["thoughts", "conclusion"],
    partial_variables={
        "format_instructions": MANY_THOUGHT_CONCLUSION_SCORING_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=MANY_THOUGHT_CONCLUSION_SCORING_OUTPUT_PARSER,
)


class LLMManyThoughtConclusionChain(Chain):
    llm: BaseLanguageModel

    conclusion_chain = Computed[Chain]
    scoring_chain = Computed[Chain]

    @computed("conclusion_chain")
    def calculate_conclusion_chain(llm: BaseLanguageModel, **kwargs):
        return LLMChain(
            llm=llm,
            prompt=MANY_THOUGHT_SUMMARY_PROMPT,
            output_parser=MANY_THOUGHT_SUMMARY_PROMPT.output_parser,
        )

    @computed("scoring_chain")
    def calculate_scoring_chain(llm: BaseLanguageModel, **kwargs):
        return LLMChain(
            llm=llm,
            prompt=MANY_THOUGHT_CONCLUSION_SCORING_PROMPT,
            output_parser=MANY_THOUGHT_CONCLUSION_SCORING_PROMPT.output_parser,
        )

    @property
    def input_keys(self) -> List[str]:
        return self.conclusion_chain.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        return ["summary"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Thought]:
        conclusion = self.conclusion_chain.predict(**inputs)
        score = self.scoring_chain.predict(conclusion=conclusion.conclusion, **inputs)
        return {"summary": Thought(discussion=conclusion.conclusion, score=score.score)}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        conclusion = await self.conclusion_chain.apredict(**inputs)
        score = await self.scoring_chain.apredict(
            conclusion=conclusion.conclusion, **inputs
        )
        return {"summary": Thought(discussion=conclusion.conclusion, score=score.score)}

    @property
    def _chain_type(self) -> str:
        return "many_thought_summarization_chain"


class LLMThoughtSummarizer(ThoughtSummarizer):
    def __init__(self, llm: BaseLanguageModel, comparer: ThoughtPairComparer):
        self.comparer = comparer
        self.comparisons: List[ThoughtPairCombination] = []
        self.chain = LLMManyThoughtConclusionChain(llm=llm)

    def _collect_comparisons(
        self, thoughts: List[Thought]
    ) -> List[ThoughtPairCombination]:
        for i, thought0 in enumerate(thoughts):
            for j, thought1 in enumerate(thoughts):
                if i <= j:
                    continue
                self.comparisons.append(self.comparer.compare(thought0, thought1))

    def _collect_contradictory_thoughts(
        self, thoughts: List[Thought]
    ) -> List[ThoughtPairCombination]:
        self._collect_comparisons(thoughts)
        contradictions = [c for c in self.comparisons if c.contradictory]
        contradictory_thoughts = list(
            set(tuple(chain(*[c.thoughts for c in contradictions])))
        )
        print(
            "THOUGHTS: ",
            len(thoughts),
            " CONTRADICTIONS: ",
            len(contradictory_thoughts),
        )
        return contradictory_thoughts

    def _summarize_non_contradictory_thoughts(self, thoughts: List[Thought]) -> Thought:
        formatted_thoughts = "\n".join(
            ["{" + str(thought) + "}" for thought in thoughts]
        )
        if len(thoughts) == 0:
            print(
                "THOUGHTS: ",
                len(thoughts),
            )
            import pdb

            pdb.set_trace()
        return self.chain({"thoughts": formatted_thoughts})["summary"]
