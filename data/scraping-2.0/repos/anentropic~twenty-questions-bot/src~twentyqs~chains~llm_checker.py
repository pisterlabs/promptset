from langchain import LLMChain, Wikipedia
from langchain.agents.react.base import ReActChain
from langchain.agents.react.wiki_prompt import EXAMPLES  # , SUFFIX
from langchain.chains import SequentialChain
from langchain.chains.base import Chain
from langchain.chains.llm_checker.prompt import (
    # CHECK_ASSERTIONS_PROMPT,
    CREATE_DRAFT_ANSWER_PROMPT,
    LIST_ASSERTIONS_PROMPT,
    REVISED_ANSWER_PROMPT,
)
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from pydantic import Extra


"""
TODO:
- the base LLM gives pretty good answers already
- Langchain LLMChecker can get it to improve with self-reflection
- no knowledge after Sept 2021, facts which changed since will get wrong answers
- we can use MRKL or ReAct to get knowledge from Wikipedia, Google, etc. which itself
  can improve some factual answers, while giving access to up-to-date info. However
  these chains also do worse on some 'common sense' questions, e.g.
  "Are camels bigger than dogs?"
- ultimately I would like to use a combination of the two, e.g. make an LLMChecker
  which uses ReAct (or MRKL, or ZeroShot + tools) to check the assertions
"""


_CHECK_ASSERTIONS_TEMPLATE = """Here is a bullet point list of assertions:
{assertions}
For each assertion, determine whether it is true or false. If it is false, explain why.\n\n"""
CHECK_ASSERTIONS_PROMPT = PromptTemplate(
    input_variables=["assertions"], template=_CHECK_ASSERTIONS_TEMPLATE
)

SUFFIX = """\nQuestion: {input}
{agent_scratchpad}"""

CHECK_ASSERTIONS_WIKI_PROMPT = WIKI_PROMPT = PromptTemplate.from_examples(
    EXAMPLES, SUFFIX, ["input", "agent_scratchpad"]
)


class LLMCheckerWithReActChain(Chain):
    """
    WIP....

    Chain for question-answering with self-verification.

    Example:
        .. code-block:: python

            from langchain import OpenAI, LLMCheckerChain
            llm = OpenAI(temperature=0.7)
            checker_chain = LLMCheckerChain(llm=llm)
    """

    """LLM wrapper to use."""
    llm: BaseLLM

    """Prompt to use when questioning the documents."""
    create_draft_answer_prompt: PromptTemplate = CREATE_DRAFT_ANSWER_PROMPT
    list_assertions_prompt: PromptTemplate = LIST_ASSERTIONS_PROMPT
    check_assertions_prompt: PromptTemplate = CHECK_ASSERTIONS_PROMPT
    revised_answer_prompt: PromptTemplate = REVISED_ANSWER_PROMPT

    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> list[str]:
        """
        Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> list[str]:
        """
        Return the singular output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: dict[str, str]) -> dict[str, str]:
        question = inputs[self.input_key]

        create_draft_answer_chain = LLMChain(
            llm=self.llm,
            prompt=self.create_draft_answer_prompt,
            output_key="statement",
            verbose=self.verbose,
        )
        list_assertions_chain = LLMChain(
            llm=self.llm,
            prompt=self.list_assertions_prompt,
            output_key="assertions",
            verbose=self.verbose,
        )
        check_assertions_chain = ReActChain(
            llm=self.llm,
            docstore=Wikipedia(),
            prompt=self.check_assertions_prompt,
            output_key="checked_assertions",
            verbose=self.verbose,
        )
        revised_answer_chain = LLMChain(
            llm=self.llm,
            prompt=self.revised_answer_prompt,
            output_key="revised_statement",
            verbose=self.verbose,
        )

        chains = [
            create_draft_answer_chain,
            list_assertions_chain,
            check_assertions_chain,
            revised_answer_chain,
        ]

        question_to_checked_assertions_chain = SequentialChain(
            chains=chains,
            input_variables=["question"],
            output_variables=["revised_statement"],
            verbose=True,
        )
        output = question_to_checked_assertions_chain({"question": question})
        return {self.output_key: output["revised_statement"]}

    @property
    def _chain_type(self) -> str:
        return "llm_checker_chain"
