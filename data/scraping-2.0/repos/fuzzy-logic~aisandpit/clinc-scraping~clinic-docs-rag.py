from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

# Find names and job titles of clinic doctors on a given web page

# @see https://python.langchain.com/docs/integrations/llms/ollama
# setup:
# ./ollama serve
# ./ollama run llama2
# run: python clinic-docs-rag.py 

# this uses the local llm web server apis once you have it running via ollma: https://ollama.ai/
llm = Ollama(
   model="llama2:13b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# VECTORDB-IZE WEB DATA
pages = ["https://www.sknclinics.co.uk/about-skn/expert-medical-team"];
print("data sourced from following web pages: ", pages)
all_splits = [];
for page in pages: 
    loader = WebBaseLoader(page)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    all_splits  = [*all_splits, *text_splitter.split_documents(data)];
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
retriever = vectorstore.as_retriever()
# Prompt
prompt = PromptTemplate.from_template(
    """Answer the question based only on the following documents: 
    {docs}
    
    
    Question: {question} """
)


# LLM Query Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)
# this uses the local llm web server apis once you have it running via ollma: https://ollama.ai/
llm = Ollama(
    model="llama2:13b",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)
chain = (
    {"docs": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

### FIRE OFF QUESTION
question = "bullet list the names and titles of doctors and nurses you can find in the document"
result = chain.invoke(question)



