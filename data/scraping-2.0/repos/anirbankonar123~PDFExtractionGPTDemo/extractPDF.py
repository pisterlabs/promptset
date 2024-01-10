from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from typing import Optional
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableLambda

model = ChatOpenAI(model_name="gpt-3.5-turbo-16k",temperature=1.8,max_tokens=500)
#model_kwargs={'top_p':0.5}
loader = PyPDFLoader("/home/anish/Downloads/IPCC_AR6_SYR_SPM.pdf")

documents = loader.load()

print(len(documents))


text=""
for doc in documents:
    text += doc.page_content

page_content_inp=text
#print(page_content_inp)
#print("####################################################")

class InvoiceInfo(BaseModel):
    """Information about Invoice"""
    key: str
    value: Optional[str]

class Invoice(BaseModel):
    """Information about Invoice"""
    invoices: List[InvoiceInfo]

class Overview(BaseModel):
    """Overview of the invoice content."""
    summary: str = Field(description="Provide a concise summary of the content.")
    keywords: str = Field(description="Provide keywords related to the content.")


overview_function = [
    convert_pydantic_to_openai_function(Overview)
]

invoice_tagging_function = [
    convert_pydantic_to_openai_function(Invoice)
]

summary_model = model.bind(
    functions=overview_function,
    function_call={"name":"Overview"}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information as summary in one paragraph from the Climate Change Report. Extract the keywords "),
    ("human", "{input}")
])

#Use LCEL to invoke the prompt with the function and the OutputParser
summary_chain = prompt | summary_model | JsonOutputFunctionsParser()

print(summary_chain.invoke({"input": page_content_inp}))
# #
#
# extraction_model = model.bind(
#     functions=invoice_tagging_function,
#     function_call={"name":"Invoice"}
# )
#
# template = """A invoice will be passed to you. Extract from it the invoice info as key value pairs.
# Consider the Discount from the final Invoice. Extract the Final Total amount"""
#
# #.
#
# prompt = ChatPromptTemplate.from_messages([
#     ("system", template),
#     ("human", "{input}")
# ])
#
# #Use LCEL to invoke the prompt with the function and the OutputParser
# extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="invoices")
#
# print(extraction_chain.invoke({"input": page_content_inp}))
#
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)
#
# splits = text_splitter.split_text(page_content_inp)
#
# def flatten(matrix):
#     flat_list = []
#     for row in matrix:
#         flat_list += row
#     return flat_list
#
#
# prep = RunnableLambda(
#     lambda x: [{"input": doc} for doc in text_splitter.split_text(x)]
# )
#
# chain = prep | extraction_chain.map() | flatten
# print(chain.invoke(doc.page_content))
