import gradio as gr
import os
#import getpass
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
os.environ['OPENAI_API_KEY'] = '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'
os.environ['ACTIVELOOP_TOKEN'] = 'eyJhbGciOiJIUzUxMiIsImlhdCI6MTY4NjU1ODk1MCwiZXhwIjoxNjg5NDk2NDM5fQ.eyJpZCI6Im1haWRhIn0.dS_b71zbnQk9REjFf2ZOCr1fWjt4w7FkfB12R90HIFIRk6q51Nq0Nl3UQTyn_o52Mi7asKLEtnsw6wDMa_0j_Q'
embeddings = OpenAIEmbeddings(disallowed_special=())
def Query(question):
  try:
    db = DeepLake(dataset_path="hub://wasiq/intt", read_only=True, embedding_function=embeddings)
    retriever = http://db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10
    model = ChatOpenAI(model_name='gpt-4') # switch to 'gpt-4'
    qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)
    chat_history = []
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(result["answer"])
    return result["answer"]
  except Exception as e:
        print(e)
# question= input("How can I help you today?")
# Query(question)


iface= gr.Interface(fn=Query, inputs= "text", outputs="text")
iface.launch()
# Query(" how can we send a beacon  with the  processor exception  details based on existing beacon code inside  try catch block in execute method of ExecutorServiceImpl.java that catches processor exceptions ? Provide a code  ")