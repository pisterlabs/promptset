from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA

my_openai_api_key = "" # Insert your OpenAI API key here

llm = ChatOpenAI( # Instantiate your favorite language model here
    openai_api_key=my_openai_api_key,
    model_name="gpt-4",
    callbacks=[StreamingStdOutCallbackHandler()],
    streaming=True,
    verbose=True, 
    temperature=0
  )

embeddings = OpenAIEmbeddings(openai_api_key=my_openai_api_key)
vectorstore = FAISS.load_local("faiss_index_oscat", embeddings)

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever = vectorstore.as_retriever(search_kwargs={'k': 6}),
    verbose=True, 
    return_source_documents=True
    )

# Modify the prompt below to your liking
prompt = """Write a self-contained IEC 61131-3 ST function block:

Create a PID controller with dynamic anti-wind up and manual control input for temperature control in an ammonium nitrates reactor.
Set point = 180.0, Kp = 50.0, Ki = 1.2, Kd = 10.0, limits between -100 and +100. 
Add a timer to only set the PID controller to automatic mode after 10 seconds.

Use pre-specified function blocks. 
Assign each used function module to a local variable, do not call it directly.
Do not include comments in the code. Do not use // or (* *).
Do not write code for the inner body of instantiated function blocks.
Use upper-case letters and underscores for function block names.
Use lower-case and upper-case letters for variable names.
Do not provide explanations, only the source code.

The generated ST Code must follow this format:

FUNCTION_BLOCK <name of the function block>
VAR_INPUT (** here define input variables **) END_VAR
VAR_OUTPUT (** here define output variables **) END_VAR
VAR (** here define internal temp variables **) END_VAR
(** here write ST code of this function block**) 
END_FUNCTION_BLOCK
"""

print("Prompt:\n", prompt)
print("-----------------------------------------------------------------------\n")
print("Answer:")

result = qa({"query": prompt})

print("The following", len(result["source_documents"]), "pages were used from the source document:\n")
pagenum = 0
for document in result["source_documents"]:
    print("-----------------------------------------------------------------------\n")
    pagenum += 1
    print("Page number:", pagenum)
    print("Page content:\n")
    print(document.page_content, "\n")
    print(document.metadata, "\n\n")