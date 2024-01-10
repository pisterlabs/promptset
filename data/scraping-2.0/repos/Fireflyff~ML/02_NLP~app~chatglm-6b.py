import gradio as gr
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileIOLoader
import os
from langchain import HuggingFacePipeline
# /Users/yingying/Desktop/pre_train_model/GanymedeNil_text2vec_large_chinese

from transformers import AutoTokenizer, AutoModel, pipeline
tokenizer = AutoTokenizer.from_pretrained("/Users/yingying/Desktop/pre_train_model/THUDM_chatglm-6b",
                                          trust_remote_code=True)
model = AutoModel.from_pretrained("/Users/yingying/Desktop/pre_train_model/THUDM_chatglm-6b",
                                  trust_remote_code=True)


class llmwebui:

    def file_upload(self, file):
        # 在这里添加文件上传的代码
        '''
        reader = PdfReader(file.name)
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        '''
        with open(file.name, "r") as f:
            raw_text = f.read()
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        # 文本分割
        texts = text_splitter.split_text(raw_text)
        # 构建embedding
        embeddings = HuggingFaceEmbeddings(model_name='/Users/yingying/Desktop/pre_train_model/GanymedeNil_text2vec_large_chinese')

        # 搜索相近的embedding
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0,
            top_p=0.,
            repetition_penalty=1.15
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        self.docsearch = FAISS.from_texts(texts, embeddings)
        self.chain = load_qa_chain(llm, chain_type="stuff")

        return "file upload ok"

    def chatbot(self, text):
        # 在这里添加聊天机器人的代码，使用输入的文本作为聊天机器人的输入，并返回答复文本
        query = text
        docs = self.docsearch.similarity_search(query)
        prompt = ""
        # 构建prompt
        for doc in docs:
            prompt = prompt + "\n" + doc.page_content

        result = self.chain.run(input_documents=docs, question=query)

        return result


myllmwebui = llmwebui()

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Column():
            result = gr.Interface(fn=myllmwebui.file_upload, inputs="file", outputs="text")

        with gr.Row():
            r = gr.Interface(fn=myllmwebui.chatbot, inputs="text", outputs="text")

demo.launch()