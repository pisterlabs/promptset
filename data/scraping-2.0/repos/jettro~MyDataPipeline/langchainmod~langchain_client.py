import logging

from dotenv import load_dotenv
from langchain import OpenAI, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter

from util import measure_time

load_dotenv()  # take environment variables from .env.

prompt_template = """Write a concise human readable summary of the following:
```
{text}
```

CONCISE SUMMARY in Dutch:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary "
    "(only if needed) with some more context below.\n"
    "----------\n"
    "{text}\n"
    "----------\n"
    "Give the new content, refine the original summary in Dutch "
    "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate(input_variables=["existing_answer", "text"], template=refine_template)


class LangchainClient:

    def __init__(self, model_name: str = "ada"):
        self.logger = logging.getLogger("langchain")
        self.model_name = model_name
        self.text_splitter = TokenTextSplitter(model_name=model_name, chunk_size=1700)
        self.llm = OpenAI(temperature=0, model_name=self.model_name)
        self.chain = load_summarize_chain(llm=self.llm,
                                          chain_type="refine",
                                          verbose=False,
                                          return_intermediate_steps=True,
                                          question_prompt=PROMPT,
                                          refine_prompt=refine_prompt)

    @measure_time
    def create_summary(self, long_text: str):
        if long_text:
            texts = self.text_splitter.split_text(long_text)
            docs = [Document(page_content=t) for t in texts]
            chain_output = self.chain({"input_documents": docs}, return_only_outputs=True)
            for step in chain_output["intermediate_steps"]:
                self.logger.info(step)
            summary = chain_output["output_text"]
        else:
            summary = ""
        return summary

    def verify_token_count(self, long_text: str):

        return self.text_splitter.split_text(long_text)
