import glob
import json
import os
import requests
import datetime
import time
import pandas as pd
from tqdm import tqdm

from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from annoy import AnnoyIndex
# import openai

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from generate import LLM


class AskWikidata:
    df = pd.DataFrame()
    local_llm = None

    def __init__(
        self,
        chunk_overlap=0,
        chunk_size=768,
        context_chunks=7,
        embedding_model_name="BAAI/bge-small-en-v1.5",
        index_trees=10,
        qa_model_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
        reranker_model_name="BAAI/bge-reranker-base",
        retrieval_chunks=64,
        cache_file=None,
    ):
        self.chunk_overlap = chunk_overlap
        self.chunk_size = chunk_size
        self.context_chunks = context_chunks
        self.embedding_model_name = embedding_model_name
        self.index_trees = index_trees
        self.qa_model_url = qa_model_url
        self.reranker_model_name = reranker_model_name
        self.retrieval_chunks = retrieval_chunks

        if not cache_file:
            emn = embedding_model_name.replace("/", "-")
            self.cache_file = f"cache-{chunk_size}-{chunk_overlap}-{emn}.json"
        else:
            self.cache_file = cache_file

        self.device = "cpu"
        if torch.cuda.is_available():
            print("CUDA available to torch.")
            self.device = "cuda"

    def setup(self):
        self.load_models()
        if not self.load_cache():
            self.read_data()
            self.create_embeds()
            self.save_cache()
        self.create_index()

    def load_models(self):
        print("Loading models...")

        self.embedding_model = HuggingFaceBgeEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant passages: ",
        )

        self.rerank_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            self.reranker_model_name
        )
        self.rerank_model.to(self.device)

        if not "https://" in self.qa_model_url:
            self.local_llm = LLM(self.qa_model_url)

    def read_data(self):
        directory_path = "./text_representations"

        texts = []
        metas = []

        files = glob.glob(os.path.join(directory_path, "*.txt"))
        print("Loading text representations...")
        for file_path in files:
            with open(file_path, "r") as file:
                texts.append(file.read())
                file_name = file_path.split("/")[-1]
                q_id = file_name.split(".")[0]
                metas.append({"source": f"https://www.wikidata.org/wiki/{q_id}"})
        print(f"  {len(files)} files loaded.")

        print("Creating chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        chunks = text_splitter.create_documents(texts, metadatas=metas)

        self.df = pd.DataFrame(columns=["id", "text", "source"])

        for i, c in enumerate(chunks):
            self.df.loc[i] = [i, c.page_content, c.metadata["source"]]

        print(f"  {len(self.df)} chunks.")

    def create_embeds(self):
        embeds = []
        print("Creating embeddings...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
            text = str(row["text"])
            embeddings = self.embedding_model.embed_documents([text])
            embeds.append(embeddings[0])
        self.df["embeddings"] = embeds

    def save_cache(self):
        print(f"Saving dataframe to {self.cache_file}...")
        # start = time.time()
        self.df.to_json(self.cache_file)
        # print(f"  {int(time.time() - start)} seconds.")

    def load_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Loading dataframe from {self.cache_file}...")
            # start = time.time()
            self.df = pd.read_json(self.cache_file)
            # print(f"  {int(time.time() - start)} seconds.")
            return True
        return False

    def create_index(self):
        print("Creating embedding index...")
        embed_dims = len(self.df.iloc[0]["embeddings"])
        self.index = AnnoyIndex(embed_dims, "angular")
        for i, e in enumerate(self.df["embeddings"]):
            self.index.add_item(i, e)
        self.index.build(self.index_trees)

    def retrieve(self, query: str) -> pd.DataFrame:
        print("Retrieving...")
        # TODO: what works better?
        # query_embed = self.embedding_model.embed_documents([query])[0]
        query_embed = self.embedding_model.embed_query(query)
        query_embed_float = [float(value) for value in query_embed]
        nns = self.index.get_nns_by_vector(
            query_embed_float, self.retrieval_chunks, include_distances=True
        )
        nns_ids = nns[0]
        nns_distances = nns[1]
        ret = self.df.iloc[nns_ids].copy()
        ret["retrieve_distance"] = nns_distances
        ret = ret.sort_values("retrieve_distance")
        return ret

    def rerank(self, query: str, df: pd.DataFrame):
        print("Reranking...")
        start = time.time()

        pairs = []
        for _, n in df.iterrows():
            pairs.append([query, n["text"]])

        with torch.no_grad():
            inputs = self.rerank_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            # TODO: do we truncate? do we loose information here?

            scores = (
                self.rerank_model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )

        df["rank"] = scores.to("cpu")
        ret = df.sort_values(by="rank", ascending=False).head(self.context_chunks)
        seconds = time.time() - start
        return ret, seconds

    def print_data(self):
        pd.set_option("display.max_rows", None)
        print(self.df)

    def context(self, df: pd.DataFrame):
        context = ""
        for index in reversed(df.index):
            row = df.loc[index]
            context += row["text"] + "\n"
        return context.replace("\n\n", "\n")

    def llm_generate(self, query: str, df: pd.DataFrame):
        context = self.context(df)
        prompt_func = None
        if "llama" in self.qa_model_url:
            prompt_func = self.llama_prompt
        elif "mistral" in self.qa_model_url:
            prompt_func = self.mistral_prompt
        else:
            raise Exception(f"unknown qa_model_name {self.qa_model_url}")

        if "huggingface.co" in self.qa_model_url:
            return self.hf_generate(query, context, self.qa_model_url, prompt_func)
        # elif "api.runpod.ai" in self.qa_model_url:
        #     return self.hf_generate(query, context, self.qa_model_url, prompt_func)
        #     return self.ask_llama_runpod(query, context)
        # return self.ask_openai(query, context)
        else:
            return self.local_generate(query, context, prompt_func)

    def llm_generate_plain(self, query: str):
        prompt_func = None
        if "llama" in self.qa_model_url:
            prompt_func = self.llama_prompt
        elif "mistral" in self.qa_model_url:
            prompt_func = self.mistral_prompt
        else:
            raise Exception(f"unknown qa_model_name {self.qa_model_url}")

        # TODO: DRY (see llm_generate)
        if "huggingface.co" in self.qa_model_url:
            return self.hf_generate(query, None, self.qa_model_url, prompt_func)
        # elif "api.runpod.ai" in self.qa_model_url:
        #     return self.hf_generate(query, context, self.qa_model_url, prompt_func)
        #     return self.ask_llama_runpod(query, context)
        # return self.ask_openai(query, context)
        else:
            return self.local_generate(query, None, prompt_func)

    def ask(self, query: str):
        retrieved = self.retrieve(query)
        reranked, _ = self.rerank(query, retrieved)
        answer = self.llm_generate(query, reranked)
        sources = set(reranked["source"])
        sources_bullet_list = "Sources:\n" + "\n".join(f"- {s}" for s in sources)
        return answer + "\n\n" + sources_bullet_list

    def system_from_context(self, context):
        system = (
            "You are answering questions for a given context. "
            + "Answer based on information from the given context only, but do not mention the context in your response. "
            + "If the answer is not in the context say that you do not know the answer. "
            + "Only give the answer, do not provide any further explanations. "
            + "Do not mention the context. "
            + "Dates and timespans will be presented to you in YYYY-MM-DD format. Interpret those. "
            + "For reference, today is the "
            + f"{datetime.date.today().strftime('%Y-%m-%d')}. "
            + "Respond with the most current information unless requested otherwise.\n"
            + f"\nCONTEXT:\n{context}"
        )
        return system

    DEFAULT_SYSTEM = (
        "You are a helpful assistent. Please answer the following question. "
    )

    def llama_prompt(self, text, system=DEFAULT_SYSTEM):
        return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{text} [/INST]"

    def mistral_prompt(self, text, system=DEFAULT_SYSTEM):
        return f"<s>[INST] {system}\n\nQUESTION: {text} [/INST]"

    def hf_generate(self, question, context, model_url, prompt_func):
        huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

        if huggingface_api_key is None:
            raise Exception("HUGGINGFACE_API_KEY is None.")

        print("Generating...")
        headers = {"Authorization": f"Bearer {huggingface_api_key}"}

        prompt = ""
        if context:
            system = self.system_from_context(context)
            prompt = prompt_func(question, system)
        else:
            prompt = prompt_func(question)

        # print(f"Sending the following prompt to {model_url}:")
        # print(prompt)
        # print(f"({len(prompt)} chars, about {int(len(prompt)/3)} tokens)")

        response = requests.post(
            model_url,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    # max is 250 https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
                    "max_new_tokens": 250,
                },
            },
        )
        # print(response.json())

        try:
            return response.json()[0]["generated_text"].replace(prompt, "").strip()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print(response)
            raise e

    def local_generate(self, question, context, prompt_func):
        if self.local_llm is None:
            raise Exception("no local llm loaded")

        print("Generating...")
        prompt = ""
        if context:
            system = self.system_from_context(context)
            prompt = prompt_func(question, system)
        else:
            prompt = prompt_func(question)

        result = self.local_llm(prompt)
        return result

    # def ask_llama_runpod(self, question, context):
    #     RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
    #     # access LLM on runpod
    #     url = f"https://api.runpod.ai/v2/llama2-7b-chat/runsync"
    #     headers = {
    #         "Authorization": RUNPOD_API_KEY,
    #         "Content-Type": "application/json",
    #     }
    #     system = self.system_from_context(context)
    #     prompt = self.llama_prompt(question, system)
    #     # print(prompt)
    #     payload = {
    #         "input": {
    #             "prompt": prompt,
    #             "sampling_params": {
    #                 "max_tokens": 1000,
    #                 "n": 1,
    #                 "presence_penalty": 0.2,
    #                 "frequency_penalty": 0.7,
    #                 "temperature": 0.0,
    #             },
    #         }
    #     }
    #     response = requests.post(url, headers=headers, json=payload)
    #     response_json = json.loads(response.text)
    #     # print(response_json)
    #     return response_json["output"]["text"][0]
    #
    # # access LLM on openai
    # def ask_openai(self, question, context):
    #     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    #     openai.api_key = OPENAI_API_KEY
    #     openai_model = "gpt-3.5-turbo"
    #     # openai_model = "gpt-4"
    #     # openai_model = "gpt-4-1106-preview"
    #     system = self.system_from_context(context)
    #     messages = [
    #         {"role": "system", "content": system},
    #         {"role": "user", "content": question},
    #     ]
    #     # print(messages)
    #     chat = openai.ChatCompletion.create(model=openai_model, messages=messages)
    #     # print(chat)
    #     reply = chat.choices[0].message.content
    #     return reply
    #
