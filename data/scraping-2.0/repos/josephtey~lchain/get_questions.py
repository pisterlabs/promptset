import faiss
import pickle
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Load the Vector Database + Langchain
index = faiss.read_index("docs.index")
with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)
store.index = index

# Setup question asking LLM
llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["text"],
    template="What is a question that the following text can answer? {text}?",
)
chain = LLMChain(llm=llm, prompt=prompt)

# Generater question for each chunk
for doc in store.index_to_docstore_id.items():
    index, id = doc
    chunk_text = store.docstore.search(id).page_content
    print(chain.run(chunk_text))
