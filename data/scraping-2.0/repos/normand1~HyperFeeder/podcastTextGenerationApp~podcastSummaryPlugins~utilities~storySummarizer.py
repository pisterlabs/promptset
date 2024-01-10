import os
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain


class StorySummarizer:
    def summarize(self, text):
        docs = [Document(page_content=text)]
        OpenAI(
            model=os.getenv("OPENAI_MODEL_SUMMARY"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS_SUMMARY")),
            temperature=0,
        )
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        result = chain.run(docs)
        return result
