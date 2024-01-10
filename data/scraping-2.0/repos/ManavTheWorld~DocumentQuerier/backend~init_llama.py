from transformers import AutoTokenizer, pipeline, logging
from constants import MODELS, REVISIONS, PATHS, SECRETS
import os
os.environ["OPENAI_API_KEY"] = SECRETS.KEYS.OPEN_AI_KEY
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext, GPTVectorStoreIndex, LLMPredictor
from constants import MODELS, REVISIONS, PATHS

def init_llm():
  model_name_or_path = MODELS.CHAT_LLMS.thebloke_13b
  model_basename = "model"

  use_triton = False

  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

  model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
          revision=REVISIONS.eight_bit_128g,
          model_basename=model_basename,
          use_safetensors=True,
          trust_remote_code=True,
          device="cuda:0",
          quantize_config=None)

  system_prompt = "You are an AI assistant that helps physicians diagnose patients. You are given a patient's symptoms and you must diagnose the patient, or answer questions related to the patient to the best of your ability."

  llm = HuggingFaceLLM(context_window=4096,
                      max_new_tokens=2048,
                      model=model,
                      tokenizer=tokenizer,
                      system_prompt=system_prompt,
                      )

  embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name=MODELS.EMBED_MODELS.sentence_transformers_all_mpnet_base_v2)
  )


  service_context = ServiceContext.from_defaults(
      chunk_size=1024,
      llm=llm,
      embed_model=embed_model
  )

  documents = SimpleDirectoryReader(PATHS.documents).load_data()
  index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
  query_engine = index.as_query_engine()
  print('Done loading index. Ready for queries.\n')
  return query_engine

def init_gpt():
  llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=512))
  service_context = ServiceContext.from_defaults(chunk_size=1024)
  documents = SimpleDirectoryReader(PATHS.documents).load_data()
  index = GPTVectorStoreIndex(documents, service_context=service_context, show_progress=True)
  query_engine = index.as_query_engine()
  return query_engine