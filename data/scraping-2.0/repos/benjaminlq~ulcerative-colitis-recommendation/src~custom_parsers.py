"""Drug OutputParser
"""
from __future__ import annotations

import re
from typing import List, TypeVar

from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field

T = TypeVar("T")

instructions = """
The output should be formatted as a list of JSON instances that conforms to the schema below.
Return an empty list if there are no drugs information inside the completion.

Here is the output schema:
```
Example 1:
[
    {"drug_name": Name of drug 1, "advantages": Summarized advantages of drug 1, "disadvantages": Summarized disadvantages of drug 1},
    {"drug_name": Name of drug 2, "advantages": Summarized advantages of drug 2, "disadvantages": Summarized disadvantages of drug 2}}
]

Example 2:
[]
"""

DRUG_FIXING = """Instructions:
--------------
{instructions}
--------------

Completion:
--------------
{completion}
--------------

Above, the Completion did not satisfy the constraints given in the Instructions.
Error:
--------------
OutputParserException
--------------

Please try again. Please only respond with an answer that satisfies the constraints laid out in the Instructions:
"""

DRUG_FIX_PROMPT = PromptTemplate.from_template(DRUG_FIXING)


class DrugOutput(BaseModel):
    """Drug Answer Schema"""

    drug_name: str = Field(description="Name of the drug")
    advantages: str = Field(description="Advantages of the drug ")
    disadvantages: str = Field(description="Disadvantages of the drug")


class DrugParser(BaseOutputParser[T]):
    """OutputParser for Drug"""

    parser: BaseOutputParser
    retry_chain: LLMChain

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        parser: BaseOutputParser[T] = PydanticOutputParser(pydantic_object=DrugOutput),
        prompt: BasePromptTemplate = DRUG_FIX_PROMPT,
    ) -> DrugParser[T]:
        """Initialize DrugParser Object from LLM.

        Args:
            llm (BaseLanguageModel): Language Model. For fixing output to correct format
            parser (BaseOutputParser[T], optional): Default Parser. Defaults to PydanticOutputParser(pydantic_object=DrugOutput).
            prompt (BasePromptTemplate, optional): Prompt for fixing output. Defaults to DRUG_FIX_PROMPT.

        Returns:
            DrugParser[T]: DrugParser
        """
        chain = LLMChain(llm=llm, prompt=prompt)
        return cls(parser=parser, retry_chain=chain)

    def parse(self, completion: str) -> List[T]:
        """Parse Text Output, fix it using LLM and return DrugOutput object

        Args:
            completion (str): input answer from LLMChain

        Returns:
            List[T]: List of DrugOutput objects
        """
        new_completion = self.retry_chain.run(
            instructions=instructions,
            completion=completion,
        )
        drugs_info = re.findall(re.compile(r"{[^{}]+}"), new_completion)
        drugs = []
        for drug in drugs_info:
            drug = self.parser.parse(drug)
            drugs.append(drug)

        return drugs

    def get_format_instructions(self) -> str:
        """Return format instructions of default parser

        Returns:
            str: Format instructions of default parser
        """
        return self.parser.get_format_instructions()

    @property
    def _type(self) -> str:
        """Chain Type"""
        return "drug_output_fixing"
