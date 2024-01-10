import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from openai import OpenAI
import pinecone
from dotenv import load_dotenv
from tqdm import tqdm


class CustomLLMChatModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_answer(self, prompt, use_sampling=False, temperature=1.0, top_p=None):
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        self.model.eval()
        with torch.no_grad():
            generation_args = {
                "max_new_tokens": 500,
                "do_sample": use_sampling,
                "temperature": temperature if use_sampling else 1.0,
                "top_p": top_p if use_sampling else None
            }
            output = self.model.generate(**model_input, **generation_args)[0]
            raw_output = self.tokenizer.decode(
                output, skip_special_tokens=True)
            return raw_output.split("### ANSWER:")[-1].strip()


class OpenAIEmbedding:
    def __init__(self, api_key, model="text-embedding-ada-002"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def text_to_vector(self, text):
        text = text.replace("\n", " ")
        res = self.client.embeddings.create(input=[text], model=self.model)
        return res.data[0].embedding


class PineconeManager:
    def __init__(self, api_key, environment, index_name):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)

    def query_index(self, vector, top_k=4):
        return self.index.query(vector=vector, top_k=top_k, include_metadata=True)


def load_environment_variables():
    load_dotenv()
    return {
        "pinecone_key": os.getenv("PINECONE_KEY"),
        "openai_key": os.getenv("OPENAI_KEY")
    }


def initialize_models(base_model, tokenizer, env_vars):
    custom_llm_model = CustomLLMChatModel(base_model, tokenizer)
    openai_embedding = OpenAIEmbedding(api_key=env_vars["openai_key"])
    pinecone_manager = PineconeManager(
        api_key=env_vars["pinecone_key"], environment="gcp-starter", index_name="document-embeddings")
    return custom_llm_model, openai_embedding, pinecone_manager


def main():
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,5,6,7"

    # Model and Tokenizer Initialization
    base_model_id = "NousResearch/Llama-2-7b-hf"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, add_bos_token=True, trust_remote_code=True)
    path_llm_model = "08_Lora_and_Rag/00_trained_lora_model/lora_finetuning/llama2-7b-AmazonVPC-finetune/checkpoint-500"
    ft_model = PeftModel.from_pretrained(base_model, path_llm_model)

    # Load environment variables and initialize models
    env_vars = load_environment_variables()
    custom_llm_model, openai_embedding, pinecone_manager = initialize_models(
        ft_model, tokenizer, env_vars)

    # Read CSV File
    test_df = pd.read_csv(
        '06_Data/Capstone_Data/documentation_qa_datasets/Final_FILTERED_TEST_Question_Answer_Pairs.csv')
    test_subset = test_df.sample(frac=1)  # Select 100% of data for full run

    for idx, row in tqdm(test_subset.iterrows(), total=test_subset.shape[0], desc="Processing Questions"):
        question = row['Question']
        question_vector = openai_embedding.text_to_vector(question)
        query_results = pinecone_manager.query_index(question_vector)
        context = ' '.join([match['metadata']['text']
                           for match in query_results['matches']])
        full_prompt = f"""
        Context: The following API reference information has been retrieved based on the user's question. Pay attention to function names, parameters, and any mentioned errors. Use this information to provide a technically accurate answer.

        Instructions: ONLY OUTPUT A ONE PARAGRAPH ANSWER.

        Retrieved API Information: {context}
        
        QUESTION: {question}
        
        ANSWER:
        """
        llm_answer = custom_llm_model.generate_answer(full_prompt)
        test_df.loc[idx, 'llm_answer'] = llm_answer

    # Save Results to New CSV
    test_df.to_csv(
        '06_Data/Capstone_Data/llm_testing_results/lora_plus_rag_testing_output.csv', index=False)


if __name__ == "__main__":
    main()
