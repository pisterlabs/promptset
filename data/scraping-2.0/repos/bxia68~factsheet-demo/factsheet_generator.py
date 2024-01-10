import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, List, Mapping, Any
import pandas as pd
import chromadb
import re
import openai
import os
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    PromptTemplate,
    LLMPredictor,
    StorageContext,
)
from llama_index.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    OpenAI,
)
from llama_index.vector_stores import ChromaVectorStore 
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms.base import llm_completion_callback
from llama_index.callbacks import (
    CallbackManager,
    LlamaDebugHandler,
)
from llama_index.schema import Node, NodeWithScore
from llama_index.response_synthesizers import get_response_synthesizer
from chromadb.utils import embedding_functions

openai.api_key = os.environ["OPENAI_API_KEY"]

def load_local_models(llm_name: str = "HuggingFaceH4/zephyr-7b-alpha", 
                    quantize=True,
                    context_window=3900,
                    max_output=256,
                    embed_name: str = "BAAI/bge-small-en-v1.5") -> CustomLLM:
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        local_llm = AutoModelForCausalLM.from_pretrained(llm_name,
                                                    device_map="auto",
                                                    quantization_config=quantization_config)
    else:
        local_llm = AutoModelForCausalLM.from_pretrained(llm_name,
                                                    device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    pipe = pipeline("text-generation", model=local_llm, tokenizer=tokenizer)
    llm_args = {
        "llm_name": llm_name,
        "context_window": context_window,
        "max_output": max_output,
    }

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embed_name)
    embed_model = HuggingFaceEmbedding(embed_name)
    return {"pipe": pipe, "llm_args": llm_args, "chroma_embed_model": sentence_transformer_ef, "embed_model": embed_model}

class FactsheetGenerator:
    def __init__(self):
        self.service_context = ServiceContext.from_defaults()
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embed_function = embedding_functions.OpenAIEmbeddingFunction()

    def use_local_model(self,
                        local_model_args,
                        debug: bool = False):
        class LocalLLM(CustomLLM):
            @property
            def metadata(self) -> LLMMetadata:
                """Get LLM metadata."""
                return LLMMetadata(
                    model_name=local_model_args["llm_args"]["llm_name"],
                    context_window=local_model_args["llm_args"]["context_window"],
                    num_output=local_model_args["llm_args"]["max_output"],
                )

            @llm_completion_callback()
            def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
                prompt_length = len(prompt)
                response = local_model_args["pipe"](prompt, max_new_tokens=local_model_args["llm_args"]["max_output"])[0]["generated_text"]

                text = response[prompt_length:]
                return CompletionResponse(text=text)

            @llm_completion_callback()
            def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
                raise NotImplementedError()
            
        self.llm = LocalLLM()

        llama_debug = LlamaDebugHandler(print_trace_on_end=debug)
        callback_manager = CallbackManager([llama_debug])
        self.service_context = ServiceContext.from_defaults(llm=self.llm,
                                                    embed_model=local_model_args["embed_model"],
                                                    callback_manager=callback_manager)
        self.embed_function = local_model_args["chroma_embed_model"]
    
    def init_vector_db(self):
        self.chroma_collection = self.chroma_client.get_or_create_collection(name="vector-db", embedding_function=self.embed_function)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection,
                                              service_context=self.service_context)

    def delete_vector_db(self):
        self.chroma_client.delete_collection(name="vector-db")

    def generate_vector_index(self, input_dir: str = "input"):
        documents = SimpleDirectoryReader(input_dir).load_data()
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(documents, 
                                                     storage_context=storage_context,
                                                     show_progress=True)

    def query(self, query: str, 
              top_n: int, 
              must_include: str):
              
        if must_include:            
            query_response = self.chroma_collection.query(
                query_texts=[query],
                n_results=top_n,
                where_document={"$contains": must_include},
                include=["documents"]
            )
        else:
            query_response = self.chroma_collection.query(
                query_texts=[query],
                n_results=top_n,
                include=["documents"]
            )

        nodes = [NodeWithScore(node=Node(text=text)) for text in query_response['documents'][0]]

        response_synthesizer = get_response_synthesizer(response_mode='compact')

        response = response_synthesizer.synthesize(
            query,
            nodes=nodes,
        )

        return response

    def generate_facts(self, 
                       strat_name: str,
                       questions: dict,
                       must_include: str = None,
                       top_n: str = 10,
                       eval: bool = False, 
                       eval_llm: LLMPredictor = None,
                       answers: dict = None):

        generated_responses = {}
        for key in questions.keys():
            response = self.query(questions[key].format(strat_name=strat_name),
                                  must_include=must_include,
                                  top_n=top_n)

            generated_responses[key] = response

        if not eval:
          return generated_responses
        else:
            df = pd.DataFrame(columns=["category", "correct source", "correct answer", "LLM review"])
            for key in generated_responses:
                df.loc[len(df)] = evaluate(key,       
                                           generated_responses[key], 
                                           questions[key],
                                           answers[key],
                                           eval_llm)
            return df

def evaluate(category: str, 
             response, 
             query: str, 
             answer: str,
             eval_llm : LLMPredictor):
    EVAL_PROMPT_TEMPLATE = (
        "Given the question below. \n"
        "---------------------\n"
        "{query_str}"
        "\n---------------------\n"
        "Decide if the following retreived context is relevant. \n"
        "\n---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Then compare the answer with the actual fact. \n"
        "\n---------------------\n"
        "answer:{answer_str}"
        "correct answer:{correct_answer_str}"
        "\n---------------------\n"
        "Answer in the following format:\n"
        "'Context is relevant: <True>\nAnswer is correct: <True>' "
        "and explain why."
    )

    DEFAULT_EVAL_PROMPT = PromptTemplate(EVAL_PROMPT_TEMPLATE) 

    if eval_llm == None:
        eval_llm = LLMPredictor(OpenAI(temperature=0, model="gpt-4"))

    def extract_eval_result(result_str: str):
        boolean_pattern = r"(True|False)"
        matches = re.findall(boolean_pattern, result_str)
        return [match == "True" for match in matches]

    result_str = eval_llm.predict(
        DEFAULT_EVAL_PROMPT,
        query_str=query,
        context_str=response.source_nodes[0].get_content(),
        answer_str=response.response,
        correct_answer_str=answer,
    )

    is_context_relevant, is_answer_correct = extract_eval_result(result_str)
    return [category, is_context_relevant, is_answer_correct, result_str]
