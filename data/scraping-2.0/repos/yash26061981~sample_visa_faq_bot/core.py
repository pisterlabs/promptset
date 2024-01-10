import os
import pandas as pd
from transformers import AutoTokenizer, pipeline
import torch
import chromadb
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CoreLLM:
    def __init__(self):
        self.gardening_path = '/tmp/gardening/'
        if not os.path.exists(self.gardening_path):
            os.mkdir(self.gardening_path)
        self.hf_cache_path = '/dbfs/tmp/cache/hf'
        os.environ['TRANSFORMERS'] = self.hf_cache_path

        self.doc = None
        self.process_qa()
        self.process()
        pass

    def process_qa(self):
        self.doc = pd.read_csv('visa_faq.csv')
        self.doc["text"] = self.doc['question'].astype(str) + " " + self.doc["answer"].astype(str)

        self.doc = self.doc[['source', 'text']]
        self.doc = self.doc.astype({"source": str})
        return

    def prepare_db(self):
        documents = [Document(page_content=self.doc.loc[r, 'text'], metadata={"source": self.doc.loc[r, 'source']}) \
                     for r in self.doc.index.values]

        chroma_version = chromadb.__version__
        print(f"chroma_version = {chroma_version}")
        #hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        # db_persist_path = f"/dbfs{gardening_path}/db"
        #client = chromadb.PersistentClient()  # (path=db_persist_path)
        # db = Chroma.from_documents(collection_name="gardening_docs", documents=documents, embedding=hf_embed, persist_directory=db_persist_path)
        #self.db = Chroma.from_documents(collection_name="gardening_docs", documents=documents, embedding=hf_embed)

        #self.db.similarity_search("dummy")

        hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.db = Chroma(collection_name="gardening_docs",
                    embedding_function=hf_embed)  # , persist_directory = db_persist_path)
        return

    def build_qa_chain(self):
        #model = "databricks/dolly-v2-3b"
        model = "databricks/dolly-v2-2-8b"
        use_bfloat = False
        if use_bfloat:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        instruct_pipeline = pipeline(model=model, torch_dtype=torch_dtype,
                                     trust_remote_code=True,
                                     device_map="auto", return_full_text=True, do_sample=False, max_new_tokens=128)

        prompt_with_context = PromptTemplate(input_variables=["question", "context"],
                                             template="{context}\n\n{question}")
        hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)
        # set verbose=True to see full prompt:
        return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt_with_context)

    def process(self):
        self.qa_chain = self.build_qa_chain()
        self.prepare_db()

        return

    def answer_question(self, question):
        print(f"Question is {question}")
        similar_docs = self.db.similarity_search(question, k=2)
        result = self.qa_chain({"input_documents": similar_docs, "question": question})
        result_html = f"\n{question}\n"
        result_html += f"answer: {result['output_text']}\n"
        for d in result["input_documents"]:
            source_id = d.metadata["source"]
            result_html += f"\ncontent:- {d.page_content},source_id:- {source_id}\n"
        #print(result_html)
        return result_html


if __name__ == '__main__':
    aq = CoreLLM()
    result_html = aq.answer_question("How to start the visa process?")
    print(result_html)
    print("Done")

