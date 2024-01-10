from langchain.chains.question_answering import load_qa_chain
import llm_model

class Question:
  def QA_Chain(self,query,data):
    llms =llm_model.Models()
    llm = llms.llm_model()
    chain = load_qa_chain(llm, chain_type = "stuff", verbose = False)
    question = query
    ans = chain.run(input_documents = data, question = question)
    print("Answer : ", ans)
    return ans
