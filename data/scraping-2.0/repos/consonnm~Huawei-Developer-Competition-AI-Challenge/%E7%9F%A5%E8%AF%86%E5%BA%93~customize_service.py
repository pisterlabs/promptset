import os
import time
import uuid
import threading
import json
from langchain.document_loaders import UnstructuredFileLoader
from mindformers import ChatGLM2Tokenizer
from mindformers.infer.infers import InferTask
from mindformers.infer.infer_config import InferConfig
from model_service.model_service import SingleNodeService
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter


# 基础镜像环境变量配置不支持推理，移除相关配置
os.unsetenv('MS_ENABLE_GE')
os.unsetenv('MS_GE_TRAIN')


def cat_weight_file(mindir_dir: str):
    for filename in os.listdir(mindir_dir):
        if '_variables' in filename:
            weight_dir = os.path.join(mindir_dir, filename)
            if len(list(weight_dir)) > 1 and not os.path.exists(f"{weight_dir}/data_0"):
                os.system(f"cat {weight_dir}/data_0_* > {weight_dir}/data_0")



class LlmService(SingleNodeService):
    def __init__(self, model_name, model_path):
        # 获取程序当前运行路径，即model文件夹所在的路径
        root = os.path.dirname(os.path.abspath(__file__))
        cat_weight_file(os.path.join(root, 'mindir'))
        self.prefill_model_path = os.path.join(root, 'mindir', 'chatglm2_6b_seq2048_bs1_full_graph.mindir')
        self.increment_model_path = os.path.join(root, 'mindir', 'chatglm2_6b_seq2048_bs1_inc_graph.mindir')
        self.ge_config_path = os.path.join(root, 'config.ini')
        self.tokenizer_path = os.path.join(root, 'checkpoint_download')
        self.knowledge_input_path = os.path.join(root, 'knowledge_input.txt')  # 知识库路径
        self.knowledge_output_path = os.path.join(root, 'knowledge_output.txt')  # 知识库路径
        self.cache_folder_path = os.path.join(root, 'embeddings')  # 知识库路径
        self.transformer_path = os.path.join(root, 'all-MiniLM-L6-v2')  # 知识库路径
        self.retriever = None  # 初始化检索器
        self.sentence_transformer = HuggingFaceEmbeddings(model_name=self.transformer_path, cache_folder=self.cache_folder_path )# 初始化句子转换器
        self.model_ready = False
        self.extraction_prompt = '【只需要给出对应选项，不需要给出原因】\n'
        self.prompt_template_z  = "已知答案相关：{}\n\n问：{}\n\n答："
        self.prompt_template= "应用相关领域知识解答\n问：{}\n\n答："
        # 非阻塞方式加载模型，防止阻塞超时
        thread = threading.Thread(target=self.load_and_warm_up)
        thread.start()
     
    def health(self):
        if self.model_ready:
            return '', 200
        else:
            return '', 404

    def load_and_warm_up(self):
        print('load and warm up start')
        lite_config = InferConfig(
            prefill_model_path=self.prefill_model_path,
            increment_model_path=self.increment_model_path,
            model_type="mindir",
            model_name="glm2",
            ge_config_path=self.ge_config_path,
            device_id=0,
            infer_seq_length=2048,
        )
        # load model
        self.tokenizer = ChatGLM2Tokenizer.from_pretrained(self.tokenizer_path)
        self.infer_model = InferTask.get_infer_task("text_generation", lite_config, tokenizer=self.tokenizer)
        # 读取和预处理知识库文件
        docs = TextLoader(self.knowledge_input_path, encoding='utf-8').load()
        documents_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
        )
        docs = documents_splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(docs, self.sentence_transformer)
        self.vector_store=self.vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .8})
        # 读取 knowledge_input.txt 文件
        with open(self.knowledge_input_path, 'r', encoding='utf-8') as file:
            self.knowledge_input_lines = [line.strip() for line in file.readlines()]
        # 读取 knowledge_output.txt 文件
        with open(self.knowledge_output_path, 'r', encoding='utf-8') as file:
            self.knowledge_output_lines = [line.strip() for line in file.readlines()]
        # warm up
        self.infer_model.infer('hello')
        print('load and warm up end')
        self.model_ready = True

    def build_prompt(self, text):
        return self.prompt_template.format(text)
    def build_prompt_z(self, text,text2):
        return self.prompt_template_z.format(text,text2)
    def _preprocess(self, data):
        process_data = {}
        for _, v in data.items():
            for file_name, file_content in v.items():
                process_data = json.loads(file_content.read())
        return process_data

    def _inference(self, data):
        query = data.get('prompt')
        query2=query
        temperature = data.get('temperature')
        top_p = data.get('top_p')
        max_tokens = data.get('max_tokens')
        choices = data.get('choices')
        if self.model_ready:
            choice_tokens = [self.tokenizer.encode(choice, add_special_tokens=False)[0] for choice in choices]
            query_ids = self.infer_model.preprocess(query, add_special_tokens=True)
            query_token_length = len(query_ids[0])
            sep="\nA"
            query2 = query2.split(sep)[0]
            similar_texts = self.vector_store.get_relevant_documents(query2)
            if(len(similar_texts)<1):
                prompt = self.extraction_prompt + self.build_prompt(query)
            else:
                # prompt = self.extraction_prompt + self.build_prompt_z(query,similar_texts[0])
                index=2
                if(len(similar_texts)<index):
                    index=len(similar_texts)
                similar_texts=similar_texts[:index]
                # 找到匹配的 'output'
                matched_outputs = []
                additional_context=[]
                # 找到这些文本片段在 knowledge_input.txt 中的行号
                line_indices = []
                for text in similar_texts:
                    if text.page_content in self.knowledge_input_lines:
                        line_indices.append(self.knowledge_input_lines.index(text.page_content))
                # 从 knowledge_output.txt 中获取相应的输出
                matched_outputs = [self.knowledge_output_lines[idx] for idx in line_indices if idx < len(self.knowledge_output_lines)]
                additional_context = ';'.join(matched_outputs)
                prompt = self.extraction_prompt+ self.build_prompt_z(additional_context,query)
            input_ids = self.infer_model.preprocess(prompt, add_special_tokens=True)
            input_token_length = len(input_ids[0])
            if input_token_length > 2048:
                print('inputs are too long, cut.')
                input_ids = input_ids[:, -2048:]

            output_ids, logits = self.infer_model.generate(
                input_ids, do_sample=False, top_k=1,
                repetition_penalty=1, eos_token_id=2, pad_token_id=0,
                is_sample_acceleration=False, streamer=None,
                top_p=top_p if top_p else 1,
                temperature=temperature if temperature else 1,
                max_length=max_tokens if max_tokens else 2048,
                return_logits=True
                )
            outputs = self.infer_model.postprocess(output_ids)
            response = outputs[0]

            logits = logits[0][:, choice_tokens]
            preds = logits.argmax(axis=-1)
            pred_choice = choices[preds[0]]

            total_token_length = len(output_ids[0])
        else:
            response = '模型编译未完成，请等待约10至20分钟'
            input_token_length = None
            total_token_length = None
            query_token_length = None
            pred_choice = None

        inference_id = str(uuid.uuid4())
        created = int(time.time())

        inference_result = {
            "id": inference_id,
            "created": created,
            "choice": pred_choice,
            "prompt_tokens": query_token_length,
            "total_tokens": total_token_length,
            "response": response
        }
        return {"result": inference_result}