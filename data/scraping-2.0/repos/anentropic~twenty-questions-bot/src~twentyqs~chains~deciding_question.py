import logging

from langchain import PromptTemplate, LLMChain
from langchain.schema import BaseOutputParser

logger = logging.getLogger(__name__)

# TODO:
# maybe add a "reason" field too for logging

template = """In a game of 20 Questions...

The oracle knows the secret subject.

The secret subject is: {subject}

The player asked: {question}
The oracle answered: Yes

Does the player now know the identity of the secret subject? (Answer only yes or no)
"""

ParsedT = bool


class IsDecidingQuestionOutputParser(BaseOutputParser[ParsedT]):
    def parse(self, text: str) -> ParsedT:
        logger.debug("IsDecidingQuestionOutputParser.parse: %s", text)
        return text.strip().lower().startswith("yes")


class IsDecidingQuestionChain(LLMChain):
    prompt = PromptTemplate(
        template=template,
        input_variables=["subject", "question"],
        output_parser=IsDecidingQuestionOutputParser(),
    )
