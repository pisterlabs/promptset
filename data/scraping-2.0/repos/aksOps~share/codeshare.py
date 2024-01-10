from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

# Define your desired data structure.
class MyData(BaseModel):
    field1: str = Field(description="Description of field1")
    field2: str = Field(description="Description of field2")

# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=MyData)
prompt = PromptTemplate(
    template="Answer the user query.\\n {format_instructions}\\n {query}\\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Initialize ConversationalRetrievalChain
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)
qa_chain = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever())

# Use ConversationalRetrievalChain and parser
_input = prompt.format_prompt(query=my_query)
output = qa_chain(_input.to_string())
parsed_output = parser.parse(output)
