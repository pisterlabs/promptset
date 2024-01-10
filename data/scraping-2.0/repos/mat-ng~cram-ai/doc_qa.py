from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub

def doc_qa (similar_texts, question):
  llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0})
  qa_chain = load_qa_chain(llm, chain_type="stuff")
  
  return qa_chain.run(input_documents=similar_texts, question=question)
