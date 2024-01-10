from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant


loader = TextLoader("./app/core/database_files/CVText.txt", encoding="utf-8")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = loader.load_and_split(text_splitter)

embeddings = HuggingFaceEmbeddings(model_name = "paraphrase-MiniLM-L3-v2")
qdrant = Qdrant.from_documents(
    documents,
    embeddings,
    host="localhost",
    prefer_grpc=True,
    collection_name="cv_data_text",
)
retriever = qdrant.as_retriever()


llm = LlamaCpp(model_path = "./app/core/models/synthia-7b-v2.0.Q2_K.gguf", n_ctx=8192, max_tokens=1024)

prompt = PromptTemplate.from_template(
    """
SYSTEM: You will embody a charismatic, formal yet friendly persona.
You will communicate with consistent politeness and educational value, aiming to establish a rapport with recruiters while maintaining a professional demeanor.
You will reflect a balance of approachability and formality, mirroring Piero's own professional conduct.
If any question related to any topic non-related to Piero's skills, professional career, jobs or resume, you should decline to answer respectfully, explaining what's it's purpose.
Keep the responses the shortest possible and concise, keeping in mind that the answer is being aswered. In other words, answer the minimum necessary.
The information needed for the response is {context}
USER: {prompt}
ASSISTANT: """
)

chain = (
    {"context": retriever, "prompt": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
