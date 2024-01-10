import abc
import time

import openai
from loguru import logger
from pydantic import BaseModel

from springtime.models.open_ai import OpenAIModel
from springtime.services.scan_service import get_chunks


class Term(BaseModel):
    term_value: str
    term_name: str


class PageOfQuestions(BaseModel):
    order: int
    value: list[str] = []


class PageOfTerms(BaseModel):
    order: int
    value: list[Term] = []


class ReportService(abc.ABC):
    @abc.abstractmethod
    def generate_questions(self, text: str) -> list[PageOfQuestions]:
        pass

    @abc.abstractmethod
    def generate_terms(self, text: str) -> list[PageOfTerms]:
        pass


MODEL = OpenAIModel.gpt3_16k


ALL_TERMS = frozenset(
    {
        "Document Name",
        "Company Overview",
        "Company Industry",
        "Document Overview",
        "Document Date",
        "Lead Arranger",
    },
)


class OpenAIReportService(ReportService):
    def generate_questions(self, text: str) -> list[PageOfQuestions]:
        acc: list[PageOfQuestions] = []
        for idx, chunk in enumerate(get_chunks(text, 30_000)):
            if questions := self.generate_questions_for_text(chunk):
                acc.append(PageOfQuestions(order=idx, value=questions))
        return acc

    def generate_terms(self, text: str) -> list[PageOfTerms]:
        acc: list[PageOfTerms] = []
        terms_needed = set(ALL_TERMS)

        for idx, chunk in enumerate(get_chunks(text, 30_000)):
            if terms := self.generate_terms_for_text(terms_needed, chunk):
                for term in terms:
                    terms_needed.remove(term.term_name)
                acc.append(PageOfTerms(order=idx, value=terms))

            if not terms_needed:
                return acc
        return acc

    def generate_questions_for_text(self, text: str) -> list[str]:
        for _attempt in range(3):
            try:
                return self._generate_questions_for_text(text)
            except openai.error.RateLimitError as e:
                logger.error(e)
                logger.error("OpenAI response for questions failed")
                logger.error("Sleeping for 10 seconds")
                time.sleep(10)
        msg = "OpenAI response for questions failed"
        raise Exception(msg)

    def _generate_questions_for_text(self, text: str) -> list[str]:
        completion = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert financial analyst AI assistant.",
                },
                {
                    "role": "user",
                    "content": "You will be given a document. Read the document and generate the top 5 most relevant/interesting questions you would want to ask about the data to better understand it for evaluating a potential investment.",
                },
                {
                    "role": "user",
                    "content": """
* Speak in the third person, e.g. do not use "you"
* Prefer proper, specific nouns to refer to entities
* Output each question on a new line. Do not output any other text.
* Use '*' for each question
""",
                },
                {"role": "user", "content": f"Document: {text}"},
            ],
            temperature=0.5,
        )
        value = completion.choices[0].message.content
        return [question.lstrip("*-").strip() for question in value.split("\n")]

    def generate_terms_for_text(self, terms_needed: set[str], text: str) -> list[Term]:
        for _attempt in range(3):
            try:
                return self._generate_terms(terms_needed, text)
            except openai.error.RateLimitError as e:
                logger.error(e)
                logger.error("OpenAI response for questions failed")
                logger.error("Sleeping for 10 seconds")
                time.sleep(10)
        msg = "OpenAI response for terms failed"
        raise Exception(msg)

    def _generate_terms(self, terms_needed: set[str], text: str) -> list[Term]:
        terms_list = "\n".join(terms_needed)

        terms = f"""
Search the document for the following terms and output their value:

{terms_list}

* Be as objective as possible. Do not output any opinions or judgments. Avoid sounding like a sales pitch
* If information for a term is not available, do not return anything for that term.
* Structure your output as Term Name | Term Value

For example,
For Lead Arranger output:

Lead Arranger | Goldman Sachs
        """

        completion = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert financial analyst AI assistant.",
                },
                {"role": "system", "content": terms},
                {"role": "user", "content": f"Document: {text}"},
            ],
            temperature=0,
        )
        response = completion.choices[0].message.content
        try:
            by_new_line = response.split("\n")
            terms = [term for line in by_new_line if (term := parse_term(line))]
            return terms

        except Exception as e:
            logger.error(e)
            logger.error("Invalid terms parsed")
            return []


IGNORE = {
    "not provided",
    "not available",
    "not specified",
    "unknown",
    "not available in the provided document.",
    "not provided in the document",
    "n/a",
    "not mentioned in the document",
}


def parse_term(value: str) -> Term | None:
    splat = value.split("|", 1)
    if len(splat) != 2:
        return None
    left, right = splat
    left = left.strip()
    right = right.strip()
    lowered = right.lower()
    if right.lower() in IGNORE or right.lower().startswith("not available"):
        return None
    return Term(term_name=left, term_value=right)
