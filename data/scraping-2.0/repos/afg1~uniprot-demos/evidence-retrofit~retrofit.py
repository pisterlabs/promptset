import os
import pickle
from tqdm import tqdm
import gradio as gr
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

docs = []
for d in tqdm(os.listdir("papers")):
    abstract = open(os.path.join("papers", d)).read().split("\n\n")[0]
    try:
        pmcid = d.strip(".txt")
        docs.append(Document(page_content=abstract, metadata={"pmcid": pmcid, "section": "abstract"}) )
    except:
        print(f"error with {d}")


passage_encoder = OpenAIEmbeddings()
if os.path.exists("ADH"):
    faiss = FAISS.load_local("ADH", passage_encoder)
else:
    faiss = FAISS.from_documents(docs, passage_encoder)
    faiss.save_local("ADH")



def make_query(query, faiss):
    docs = faiss.max_marginal_relevance_search(query, k=10)
    result = "".join(f"{doc.metadata['pmcid']}: {doc.page_content}\n" for doc in docs)
    return result


visualisation = gr.Blocks()

with visualisation:
    gr.Markdown(
        "Search queries about linc00174"
    )

    with gr.Row():
        query_input = gr.Textbox(label="Query")
        search_button = gr.Button(value="Run...")


    with gr.Row():
        result = gr.Textbox(label="Result")
    
    query_input.submit(lambda x: make_query(x, faiss),inputs=query_input,outputs=[result])
    search_button.click(lambda x: make_query(x, faiss),inputs=query_input,outputs=[result])
    


visualisation.queue(concurrency_count=1)
visualisation.launch(server_name="0.0.0.0", enable_queue=True, server_port=7860)
