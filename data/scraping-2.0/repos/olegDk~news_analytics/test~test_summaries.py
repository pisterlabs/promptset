import os
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_llm = ChatOpenAI(
    temperature=1,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
)

question_prompt_template = """
                  Please provide a summary of the following text.
                  TEXT: {text}
                  SUMMARY:
                  """

question_prompt = PromptTemplate(
    template=question_prompt_template, input_variables=["text"]
)

refine_prompt_template = """
              Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
              """

refine_prompt = PromptTemplate(
    template=refine_prompt_template, input_variables=["text"]
)

refine_chain = load_summarize_chain(
    chat_llm,
    chain_type="refine",
    question_prompt=question_prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
)

doc1 = Document(page_content="Hello, world!")
doc2 = Document(page_content="How are you doing?")

docs = [doc1, doc2]

refine_outputs = refine_chain({"input_documents": docs})
