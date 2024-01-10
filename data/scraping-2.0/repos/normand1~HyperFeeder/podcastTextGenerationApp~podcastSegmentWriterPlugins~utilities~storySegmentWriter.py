import os
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


class StorySegmentWriter:
    def writeSegmentFromSummary(self, storySummary):
        llm = OpenAI(
            model=os.getenv("OPENAI_MODEL_SUMMARY"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS_SUMMARY")),
            temperature=0.3,
        )
        templateString = os.getenv("SEGMENT_WRITER_STRING")
        prompt = PromptTemplate(
            input_variables=["storySummary"],
            template=templateString,
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(storySummary)
