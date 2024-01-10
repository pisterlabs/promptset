import torch
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain import LLMChain, HuggingFacePipeline, PromptTemplate
from post_process import *
import warnings
import pandas as pd
import os
warnings.filterwarnings("ignore")



torch.backends.cuda.matmul.allow_tf32 = True



cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)



name = "meta-llama/Llama-2-7b-chat-hf"
token = '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'
tokenizer = AutoTokenizer.from_pretrained(name, use_auth_token=token,use_fast=True)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.float16,
    bnb_8bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    name,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True, 
    use_auth_token=token,
    torch_dtype=torch.float16,
    load_in_8bit=True,
    use_flash_attention_2=True,
)


model = model.to_bettertransformer()


file_name = "bot_followup_files/empty_df.csv"
orig = pd.read_csv(file_name)
p_df = orig[orig.text.isna()]



passages = p_df.prompt.to_list()
max_seq_len = max(len(tokenizer.encode(p)) for p in passages)


max_batch_size = 2
prompt_batches = [passages[i:i + max_batch_size] for i in range(0, len(passages), max_batch_size)]
index_batches = [p_df.index[i:i + max_batch_size].tolist() for i in range(0, len(passages), max_batch_size)]
assert len(prompt_batches[0]) == max_batch_size



pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    # max_length=max_seq_len + 350,
    max_new_tokens = 300,
    # min_length=max_seq_len + 250, 
    min_new_tokens = 200,
    do_sample=True,
    top_k=20,
    top_p=0.9, 
    temperature=0.1,  
    num_return_sequences=1,
    repetition_penalty=1.5,
    eos_token_id=tokenizer.eos_token_id
)

# llm = HuggingFacePipeline(pipeline=pipeline)
# template = """{text}"""
# prompt = PromptTemplate(template=template, input_variables=["text"])
# llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)





def post_process_(responses):
    res = []
    for i in range(len(responses)):
        new_text = " ".join(responses[i][0]["generated_text"].split("[/INST]")[-1].split('\n')[1:]).strip()
        new_text = count_words_complete_sentences(filter_utf8(new_text)).strip()
        new_text = re.sub(r'^[^a-zA-Z0-9]+', '', new_text)
        res.append(new_text)
    return res



model.eval()
with torch.no_grad():
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        for batch, idx_batch in tqdm(zip(prompt_batches, index_batches), total=len(prompt_batches)):
            responses = pipeline(batch)
            texts = post_process_(responses)
            print(file_name, idx_batch)
            for i in range(len(idx_batch)):
                orig.at[idx_batch[i], "text"] = texts[i]
            orig.to_csv(file_name, index=False)
