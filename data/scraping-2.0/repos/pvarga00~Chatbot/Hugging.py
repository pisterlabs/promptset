# Orig: https://python.langchain.com/docs/integrations/llms/huggingface_hub
# https://python.langchain.com/docs/use_cases/chatbots
# https://python.langchain.com/docs/use_cases/more/code_writing/
# https://huggingface.co/bigcode/starcoder


# pip install huggingface_hub playwright beautifulsoup4
# pip install pytest-playwright


from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer

# Load HTML
loader = AsyncChromiumLoader(["https://python.langchain.com/docs/use_cases/web_scraping"])  # (["https://www.wsj.com"])
html = loader.load()

question = "Who won the FIFA World Cup in the year 1994?"
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

repo_id = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(html,tags_to_extract=["span"])

# Result
docs_transformed[0].page_content[0:500]

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64},
    huggingfacehub_api_token="API_KEY"  ## Replace me
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))

