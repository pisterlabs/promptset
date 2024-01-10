import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM
# checkpoint = "microsoft/DialoGPT-large"
checkpoint = "/Users/yingying/Desktop/pre_train_model/gpt2"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
import torch


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
        embeddings = HuggingFaceEmbeddings()

        # 搜索相近的embedding
        self.docsearch = FAISS.from_texts(texts, embeddings)
        self.step = 0
        self.chat_history_ids = None

        return f"文档内容: {raw_text}"

    def chatbot(self, text):
        # 在这里添加聊天机器人的代码，使用输入的文本作为聊天机器人的输入，并返回答复文本
        query = text
        docs = self.docsearch.similarity_search(query)
        prompt = ""
        # 构建prompt
        for doc in docs:
            prompt = prompt + "\n" + doc.page_content
        prompt = prompt + "\nAnswer the question based on known information:{}".format(query)
        encoded_input = tokenizer.encode(prompt)
        answer_len = len(encoded_input)
        tokens_tensor = torch.tensor([encoded_input])

        stopids = tokenizer.convert_tokens_to_ids(["."])[0]
        past = None
        for i in range(100):
            with torch.no_grad():
                # gpt2的参数："n_embd": 768, "n_head": 12, "n_layer": 12, "vocab_size": 50257
                # "n_ctx"(所能允许的文本最大长度): 1024, "n_positions"(通常与 n_positions 相同): 1024
                output, past = model(tokens_tensor, past_key_values=past, return_dict=False)
                # past_key_values 保存了上次迭代过程中的 key 和 value（attention 运算中的键值对）用于加速运算
                # past_key_values: ((K, Q)) * 12,
                # 因此past_key_values 的结构为 (12，2，(batch_size, num_head, sql_len, head_features))
                # 即 (12，2，(1, 12, sql_len, 64)) --> sql_len 为 encoded_input 的 length

            token = torch.argmax(output[..., -1, :])

            encoded_input += [token.tolist()]

            if stopids == token.tolist():
                break
            tokens_tensor = token.unsqueeze(0)

        sequence = tokenizer.decode(encoded_input[answer_len:])

        return sequence


myllmwebui = llmwebui()

with gr.Blocks() as demo:
    with gr.Column():
        with gr.Column():
            result = gr.Interface(fn=myllmwebui.file_upload, inputs="file", outputs="text")

        with gr.Row():
            r = gr.Interface(fn=myllmwebui.chatbot, inputs="text", outputs="text")

demo.launch(share=True)