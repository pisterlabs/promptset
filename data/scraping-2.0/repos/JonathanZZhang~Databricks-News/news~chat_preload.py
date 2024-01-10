# 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
import os
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain


class news_tag:
    def __init__(self) -> None:
        self.load_api_key()
        self.model()
    def load_api_key(self):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    

    def model(self):
       repo_id = "fabiochiu/t5-base-tag-generation"
       print("loading model, may take a while...")
       self.llm_tags = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0})
    
    def create_tag(self, news_path):
        # with open(news_path) as f:
        #     content = f.readlines()
        content = news_path
        template = """article: {article}."""
        prompt = PromptTemplate(template=template, input_variables=["article"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm_tags)
        article = content
        return llm_chain.run(article)

    
if __name__ == "__main__":
    tag = news_tag()
    print(tag.create_tag("test.txt"))
#  news_preload()
    # print(test_tag.create_tags("test.txt"))

       












