from langchain.chains import RetrievalQA

from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from vector_store import load_vector,create_vector_store
#from IPython.display import display, Markdown, Latex
import os


def llm_process_query(query,vector_store):
    #create custom prompt for your use case
    
    prompt_template="""You are Papery, an AI paper assistant that answer to user question about the trend or data about research paper based on context.
    Use the following pieces of context (title keywords) to answer the users question.

    Example conversation (not real answer):
    User : What is the popular topic
    Bot: Based on the data, it's about AI topic
    ----------------
    Context :
    {context}

    User : {question}

    """

    QA_CHAIN_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context"]
    )


    chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT,"verbose": True}

    llm=Cohere(model="command")

    retriever = vector_store.as_retriever()
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
        verbose=True
    )
    result = chain({"query":query})

    return result

#print your results with Markup language
def print_result(result,query):
  print(result)
  output_text = f"""### Question:
  {query}
  ### Answer:
  {result['result']}
  ### Sources:
  {result['source_documents']}
  ### All relevant sources:
  {' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
  """
  return(output_text)


if __name__ == "__main__":
   vector_store = load_vector('./dags/services/vector_store_folder')
   print(llm_process_query('What is the main study or topic of title that are in scopus 2015-2013',vector_store))