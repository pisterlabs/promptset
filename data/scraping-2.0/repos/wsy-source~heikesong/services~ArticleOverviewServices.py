from langchain import PromptTemplate

from llm.llm import llm
from langchain.chains import LLMChain
from prompt.ChatPrompt import Article_Prompt
from prompt.Translate_Prompt import Translate_Prompt


class ArticleOverviewServices:
    @classmethod
    def analysis_article(cls, content: str):
        prompt = PromptTemplate.from_template(Article_Prompt)
        llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
        return llm_chain.run(content)

    @classmethod
    def translate_article(cls, content: str):
        prompt = PromptTemplate.from_template(Translate_Prompt)
        chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
        return chain.run(content)
