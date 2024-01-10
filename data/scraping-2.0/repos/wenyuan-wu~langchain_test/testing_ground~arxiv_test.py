from langchain.retrievers import ArxivRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()
retriever = ArxivRetriever(load_max_docs=2)
docs = retriever.get_relevant_documents(query='1605.08386')

print(docs[0].metadata)  # meta-information of the Document
print(docs[0].page_content[:400])  # a content of the Document

model = ChatOpenAI(model_name='gpt-3.5-turbo') # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

# questions = [
#     "What are Heat-bath random walks with Markov base?",
#     "What is the ImageBind model?",
#     "How does Compositional Reasoning with Large Language Models works?",
# ]
# chat_history = []
#
# for question in questions:
#     result = qa({"question": question, "chat_history": chat_history})
#     chat_history.append((question, result['answer']))
#     print(f"-> **Question**: {question} \n")
#     print(f"**Answer**: {result['answer']} \n")


questions = [
    "What are Heat-bath random walks with Markov base? Include references to answer.",
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")

