from typing import Dict, List, Callable, Any
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLanguageModel
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from prompt_breeder.prompts.string import StringTaskPrompt
from langchain.output_parsers.pydantic import PydanticOutputParser
from pydantic import BaseModel


BASE_TEMPLATE = PromptTemplate.from_template("{task_prompt} {query}")
CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("{task_prompt}"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)
PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=BASE_TEMPLATE,
    conditionals=[(is_chat_model, CHAT_TEMPLATE)],
)


class Answer(BaseModel):
    workings: str
    final_answer: int


output_parser = PydanticOutputParser(pydantic_object=Answer)  # type: ignore


class NiaveContainsCorrectAnswer(LLMChain):
    dataset: List[Dict[str, str]]
    data_aggfunc: Callable[[List[Any], List[float]], float] = lambda ds, fs: sum(
        fs
    ) / len(ds)

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, **kwargs):
        return cls(
            llm=llm,
            prompt=PROMPT_SELECTOR.get_prompt(llm),
            **kwargs,
        )

    def get_one(self, prompt: StringTaskPrompt, datum, **kwargs):
        return {
            "query": datum["question"],
            "answer": datum["answer"],
            "response": self.run(
                {"task_prompt": str(prompt), "query": datum["question"]}, **kwargs
            ),
            "correct": self._answer_isin_completion(
                datum["answer"],
                self.run(
                    {"task_prompt": str(prompt), "query": datum["question"]}, **kwargs
                ),
            ),
        }

    def get_all(self, prompt: StringTaskPrompt, **kwargs):
        return [self.get_one(prompt, d) for d in self.dataset]

    @staticmethod
    def _answer_isin_completion(answer: str, completion: str) -> bool:
        # return answer == completion
        return answer in completion

    def _score_one(
        self, prompt: StringTaskPrompt, datum: Dict[str, str], **kwargs
    ) -> float:
        return float(
            self._answer_isin_completion(
                datum["answer"],
                self.run(
                    {"task_prompt": str(prompt), "query": datum["question"]}, **kwargs
                ),
            )
        )

    async def _ascore_one(
        self, prompt: StringTaskPrompt, datum: Dict[str, str], **kwargs
    ) -> float:
        return float(
            self._answer_isin_completion(
                datum["answer"],
                await self.arun(
                    {"task_prompt": str(prompt), "query": datum["question"]}, **kwargs
                ),
            )
        )

    def score(self, prompt: StringTaskPrompt, **kwargs) -> float:
        if str(prompt) == "":
            return 0.0
        return self.data_aggfunc(
            self.dataset, [self._score_one(prompt, d) for d in self.dataset]
        )

    async def ascore(self, prompt: StringTaskPrompt, **kwargs) -> float:
        return self.data_aggfunc(
            self.dataset, [await self._ascore_one(prompt, d) for d in self.dataset]
        )


def create_gsm8k_fitness(llm, kind: str = "train", n_samples: int = 50, **kwargs):
    from langchain.output_parsers.regex import RegexParser
    from datasets import load_dataset  # type: ignore

    dataset = load_dataset("gsm8k", "main")
    data = dataset[kind][0:n_samples]
    parser = RegexParser(regex=r"### (.*)", output_keys=["output"])
    data = [
        {"question": a, "answer": parser.parse(b)["output"]}
        for (a, b) in zip(data["question"], data["answer"])
    ]
    return NiaveContainsCorrectAnswer.from_llm(llm=llm, dataset=data, **kwargs)
