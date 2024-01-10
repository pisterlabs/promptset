import logging
import re

from langchain import PromptTemplate, LLMChain
from langchain.schema import BaseOutputParser

from twentyqs.types import Answer

logger = logging.getLogger(__name__)

# possibly there should be an "unanswerable" response too
prefix = """You are a chatbot playing a question answering game with a human.

The human will ask you yes/no questions about a subject.

The question must be accurately with one of four acceptable answers (nothing else):
- Yes: if the subject has the property asked in the question
- No: if the subject does not have the property asked in the question
- Sometimes: if the subject may or may not have the property asked about depending on the time of day
- I don't know: if you don't know the answer

For this game we use special definitions of the words "animal", "mineral" and "vegetable":
Define everything as being either Animal (if it is, or was, alive but not a vegetable) Vegetable (if it grows but is not an animal) or Mineral (if it isn't alive, doesn't grow and comes from the ground).
For example: "the Eiffel Tower" would be categorized as a mineral, as it is a non-living object that is made from metal and other materials that were extracted from the earth.
Another example: If the target is "The Great Barrier Reef". The Great Barrier Reef is predominantly made up of living organisms, primarily coral polyps, which are animals. However, the reef also contains some mineral structures and sedimentary deposits. So, it could be considered as a combination of animal, mineral, and possibly vegetable (in the form of algae) elements.
So if asked "Is it animal?" it would be correct to answer "Yes", if asked "Is it mineral?" it would be correct to answer "Yes", and if asked "Is it vegetable?" it would be correct to answer "No".

Today's date is: {today}

Now we are ready to play the game.

Use the following format:
"""

examples = [
    """Subject: Albert Einstein
Question: is it alive?
Thought: Albert Einstein died in 1955. Albert Einstein is not alive.
Answer: No""",
    """Subject: Albert Einstein
Question: is it animal?
Thought: Albert Einstein was alive but was not a vegetable.
Answer: Yes""",
    """Subject: Albert Einstein
Question: is it yellow?
Thought: Albert Einstein does not have a specific colour. His skin was predominantly 'flesh' colour.
Answer: No""",
    """Subject: The Brooklyn Bridge
Question: is it mineral?
Thought: It is a non-living object that is made from metal and other materials that were extracted from the earth.
Answer: Yes""",
    """Subject: Venus
Question: is it visible?
Thought: Venus is visible in the sky at night, but not during the day.
Answer: Sometimes""",
    """Subject: God
Question: does it exist?
Thought: The answer is unknowable.
Answer: I don't know""",
]

splitter_re = re.compile(r"(Thought|Answer)\:\s*(?P<value>.*)")

AnswerT = Answer | str | None
ParsedT = tuple[AnswerT, str | None]


def _get_matched_value(unparsed: str) -> str | None:
    match = splitter_re.match(unparsed)
    if match:
        return match.groupdict()["value"] or None
    return None


class AnswerQuestionOutputParser(BaseOutputParser[ParsedT]):
    def parse(self, text: str) -> ParsedT:
        logger.debug("AnswerQuestionOutputParser.parse: %s", text)
        thought, answer = text.rsplit("\n", 2)
        thought_ = _get_matched_value(thought)
        answer_: AnswerT = _get_matched_value(answer)
        if answer_:
            for a in Answer:
                if answer_.startswith(a.value):
                    answer_ = a
                    break
        return answer_, thought_


class AnswerQuestionChain(LLMChain):
    prompt = PromptTemplate.from_examples(
        examples=examples,
        suffix=("Subject: {subject}\n" "Question: {question}\n"),
        prefix=prefix,
        input_variables=["today", "subject", "question"],
        output_parser=AnswerQuestionOutputParser(),
    )
