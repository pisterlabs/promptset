from typing import Any

from langchain import PromptTemplate
from langchain.document_loaders import AzureBlobStorageFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool
from llm.llm import llm
from langchain.chains.summarize import load_summarize_chain
from prompt.QuestionAndAnswerPrompt import QUESTION_AND_ANSWER_PROMPT
import os
from langchain.chains import LLMChain


class QuestionAndAnswer:

    @classmethod
    def answer_question_with_data(cls, question, content):
        prompt = PromptTemplate.from_template(QUESTION_AND_ANSWER_PROMPT)
        chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
        result = chain.run(question=question, input_documents=content)
        return result
