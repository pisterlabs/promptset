
import utils
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

PROMPT="""Use the following pieces of context to answer the question at the end.
If the question is not directly related to the context, just say that you don't know, don't try to make up an answer. 
If not enough information is available in the context, just say that you don't know, don't try to make up an answer.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
In any case, please do not leverage your own knowledge.

{context}

Question: {question}
Helpful Answer:"""

class QAChainBase:

  chain = None

  @staticmethod
  def build(llm, retriever):

    if QAChainBase.chain is None:
    
      # build prompt
      prompt = PromptTemplate(input_variables=['context', 'question'], template=PROMPT)
      
      # build chain
      print('[chain] building basic retrieval chain')
      chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={ 'prompt': prompt },
      )
      utils.dumpj({
        'llm': chain.combine_documents_chain.llm_chain.prompt.template,
      }, 'chain_templates.json')
      QAChainBase.chain = chain

    # done
    return (QAChainBase.chain, 'query')
