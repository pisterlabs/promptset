from torch import cuda, bfloat16
import torch
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import os
import cv2
import argparse
import sys
TOKEN = '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' # Change the token as it might be out-dated
class ModelInference:
    def __init__(self, model_id, device, hf_auth = TOKEN):
        self.model_id = model_id
        self.device = device
        # begin initializing HF items, you need an access token
        self.hf_auth = hf_auth
        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        self.bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )
        self.model_config = transformers.AutoConfig.from_pretrained(
                self.model_id,
                use_auth_token=self.hf_auth
        )

    def load_model(self):
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                config=self.model_config,
                quantization_config=self.bnb_config,
                device_map='auto',
                use_auth_token=self.hf_auth
        )
        # enable evaluation mode to allow model inference
        self.model.eval()
        print(f"Model loaded on {self.device}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id,
                use_auth_token=self.hf_auth
        )
        stop_list = ['\nHuman:', '\n```\n']
        stop_token_ids = [self.tokenizer(x)['input_ids'] for x in stop_list]
        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
    # define custom stopping criteria object
        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for stop_ids in stop_token_ids:
                    if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                        return True
                return False


        self.stopping_criteria = StoppingCriteriaList([StopOnTokens()])
        self.__create_pipeline()
    
    def __create_pipeline(self):
        generate_text = transformers.pipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=True,  # langchain expects the full text
                task='text-generation',

                # we pass model parameters here too
                stopping_criteria=self.stopping_criteria,  # without this model rambles during chat
                temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
                max_new_tokens=512,  # max number of tokens to generate in the output
                repetition_penalty=1.1  # without this output begins repeating
        )

        #res = generate_text("Explain me the similarites between Data Lakehouse and Data Warehouse.")
        #print(res[0]["generated_text"])
        self.llm = HuggingFacePipeline(pipeline=generate_text)

        # checking again that everything is working fine
        self.llm(prompt="Explain me the difference between Data Lakehouse and Data Warehouse.")

    def get_embeddings(self, doc_path):
        loader = PyPDFDirectoryLoader(doc_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        all_splits = text_splitter.split_documents(docs)
        if os.path.isfile(doc_path):
            print(f" \n all splits type {type(all_splits)} and value \n {all_splits} \n\n ")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": "cuda"}
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        # storing embeddings in the vector store
        self.vectorstore = FAISS.from_documents(all_splits, embeddings)

    def inference(self,query):
        chain = ConversationalRetrievalChain.from_llm(self.llm, self.vectorstore.as_retriever(), return_source_documents=False)
        chat_history = []
        print("\n*******Complutation COMPLETE********\n")

        # query = "Can I share data with third parties?"
        result = chain({"question": query, "chat_history": chat_history})

        print(result['answer'])



if __name__ == '__main__':

    model_id = 'meta-llama/Llama-2-7b-chat-hf'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--doc_path')
    args = parser.parse_args()
    
    doc_path = args.doc_path
    #Doc Path
    # doc_path = "JamesBond"#"ProcDoc" #"JamesBond"
    # querys = ["So what does james bond do in casino royal?",
    #           "Who was bond's love interest in each of the books given in to you?"
    # ]
    t1 = cv2.getTickCount()
    model_inf = ModelInference(model_id, device)
    model_inf.load_model()
    model_inf.get_embeddings(doc_path)
    t2 = cv2.getTickCount()
    time = (t2-t1)/ cv2.getTickFrequency()
    print(f"total time taken for prep {time} sec!!!")
    # for query in querys:
    times = []
    print("Hello this is Info Aggregator!!!\nAsk anything regarding ")
    while(1):
        query = input("Ask away your questions!!!\n")
        if query == "quit":
            break
        t1 = cv2.getTickCount()
        model_inf.inference(query)
        t2 = cv2.getTickCount()
        time = (t2-t1)/ cv2.getTickFrequency()
        times.append(time)
        # print(f"total time taken {time} sec!!!")
        print("\n\n\n")
    print(f"completed!!! times taken are {times}")