import openai
import time
import os
import random
from cprint import *
import math
import tiktoken
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch

############################################################################################################
############################################################################################################

# =========================== begin of local chat keys ===========================
key_pool = []
f = open("./utils/available_keys.txt", "r")
lines = f.readlines()
f.close()
for line in lines:
    key_pool.append(line.strip())
key_num = len(key_pool)

def chat_api(temperature=0.3, messages=[]):
    try_limit = 2000
    try_num = 0
    try_start = random.randint(0, key_num-1)
    while try_num < try_limit:
        try:
            # print("Try number:", try_num)
            if try_num % key_num == 0 and try_num != 0:
                cprint.err("sleeping...")
                time.sleep(random.uniform(4, 6))
            openai.api_key = key_pool[(try_start+try_num) % key_num]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature,
                max_tokens=256,
                top_p=1.0,
            )
            return response["choices"][0]["message"]["content"]
        except:
            pass
        try_num += 1
    raise Exception("API key exhausted")
# =========================== end of local chat keys ===========================

############################################################################################################
############################################################################################################

# =========================== begin of MPT API ===========================
model = tokenizer = None

# Change to the path to MPT model
if os.environ.get("MODEL") == "MPT":
    model_path = "./mpt_model"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,config=config, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

def mpt_api(temperature=0.3, messages=[]):
    stop_token_ids = tokenizer.convert_tokens_to_ids(['<|endoftext|>', '<|im_end|>'])
    
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False
    
    generate_kwargs = {
        'max_new_tokens': 512,
        'temperature': temperature,
        'top_p': 0.5,
        'use_cache': True,
        'do_sample': True,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.eos_token_id,
        'stopping_criteria': StoppingCriteriaList([StopOnTokens()]),
    }

    conversation = mpt_format_history(messages)
    input_ids = tokenizer(conversation, return_tensors='pt').input_ids
    input_ids = input_ids.to(model.device)
    # also stream to stdout
    torch.cuda.synchronize()
    gkwargs = {**generate_kwargs, 'input_ids': input_ids}
    # this will stream to stdout, but we need to keep track of the output_ids for saving history
    outputs = model.generate(**gkwargs)
    torch.cuda.synchronize()
    new_tokens = outputs[0, len(input_ids[0]):]
    assistant_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return assistant_response.strip()


def mpt_confidence(temperature=0.3, messages=[]):
    stop_token_ids = tokenizer.convert_tokens_to_ids(['<|endoftext|>', '<|im_end|>'])
    
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    generate_kwargs = {
        'max_new_tokens': 512,
        'temperature': temperature,
        'top_p': 1.0,
        'top_k': 0,
        'use_cache': True,
        'do_sample': True,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.eos_token_id,
        'stopping_criteria': StoppingCriteriaList([StopOnTokens()]),
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    conversation = mpt_format_history(messages)
    last_qst = get_last_question(messages)
    
    input_ids = tokenizer(conversation, return_tensors='pt').input_ids
    input_ids = input_ids.to(model.device)
    # also stream to stdout
    torch.cuda.synchronize()
    gkwargs = {**generate_kwargs, 'input_ids': input_ids}
    # this will stream to stdout, but we need to keep track of the output_ids for saving history
    outputs = model.generate(**gkwargs)
    torch.cuda.synchronize()
    new_tokens = outputs.sequences[0, len(input_ids[0]):]

    ori_res_w_special = tokenizer.decode(new_tokens, skip_special_tokens=False)
    ori_res_wo_special = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"--- original answer ---\n{ori_res_wo_special}")
    # extract the target answer
    tgt_ans = extract_answer(last_qst, ori_res_wo_special.strip())
    if "N/A" in tgt_ans:
        cprint.err("N/A in target answer! return -1 confidence!")
        return (ori_res_wo_special.strip(), "N/A", -1)
    
    # Prepare to retrieve the confidence
    raw_probs = torch.stack(outputs.scores, dim=1).softmax(-1).squeeze() # [len, vocab]
    valid_scores = torch.gather(raw_probs, 1, new_tokens[:, None]).squeeze().cpu().tolist() # [len]
        
    probs = get_prob(ori_res_w_special, tgt_ans, valid_scores, tokenizer)
    if probs == -1:
        cprint.err("The extract answer fail! the confidence return as -1!")
        cprint.err(ori_res_wo_special)
        cprint.err(tgt_ans)
    prob = float(math.prod(probs))
    return (ori_res_wo_special.strip(), tgt_ans, prob)


def mpt_format_history(messages):
    def replace_nl(text):
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        return text.strip()
    ret = ""
    for msg in messages:
        if msg["role"] == "system":
            text = replace_nl(msg["content"])
            ret += f"<|im_start|>{text}<|im_end|>\n\n"
        elif msg["role"] == "user":
            text = replace_nl(msg["content"])
            ret += f"<|im_start|>user\n{text}<|im_end|>\n\n"
        elif msg["role"] == "assistant":
            text = replace_nl(msg["content"])
            ret += f"<|im_start|>assistant\n{text}<|im_end|>\n\n"
    ret += "<|im_start|>assistant\n"
    return ret
# =========================== end of MPT API ===========================

############################################################################################################
############################################################################################################

# =========================== begin of Davinci API ===========================
davinci_enc = tiktoken.encoding_for_model("text-davinci-003")

def davinci_confidence(temperature=0.3, messages=[]) -> str:
    prompt = chat_to_davinci_formatter(messages)
    last_qst = get_last_question(messages)
    try_limit = 2000
    try_num = 0
    try_start = random.randint(0, key_num-1)
    while try_num < try_limit:
        try:
            if try_num % key_num == 0 and try_num != 0:
                time.sleep(random.uniform(4, 6))
                cprint.err("sleeping...")
            openai.api_key = key_pool[(try_start+try_num) % key_num]
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=temperature,
                max_tokens=256,
                top_p=1.0,
                logprobs=0,
            )
            # Get the original response
            ori_res:str = response["choices"][0]["text"]
            print(f"--- original answer ---\n{ori_res}")
            # extract the target answer
            tgt_ans = extract_answer(last_qst, ori_res.strip())
            if "N/A" in tgt_ans:
                return (ori_res.strip(), "N/A", -1)
            log_probs = response["choices"][0]["logprobs"]["token_logprobs"]
            logprobs = get_prob(ori_res, tgt_ans, log_probs, davinci_enc)
            prob = math.pow(math.e, sum(logprobs))
            return (ori_res.strip(), tgt_ans, prob)
        except:
            pass
        try_num += 1
    raise Exception("API key exhausted")

def chat_to_davinci_formatter(messages):
    def replace_nl(text):
        while "\n\n" in text:
            text = text.replace("\n\n", "\n")
        return text.strip()
    ret = ""
    for msg in messages:
        if msg["role"] == "system":
            text = replace_nl(msg["content"])
            ret += f"- System:\n{text}\n\n"
        elif msg["role"] == "user":
            text = replace_nl(msg["content"])
            ret += f"- User:\n{text}\n\n"
        elif msg["role"] == "assistant":
            text = replace_nl(msg["content"])
            ret += f"- Assistant:\n{text}\n\n"
    ret += "- Assistant:\n"
    return ret
# =========================== end of Davinci API ===========================

############################################################################################################
############################################################################################################

# Other functions
def is_sublist(a, b):
    n, m = len(a), len(b)
    if n < m:
        return -1
    for i in range(n - m + 1):
        if a[i:i + m] == b:
            return i
    return -1

# Get the probability of subwords
def get_prob(response, target, scores, enc):
    res_e = enc.encode(response)
    res_t = enc.encode(target)
    if len(res_e) != len(scores):
        return [-1]
    if is_sublist(res_e, res_t) == -1:
        res_t = enc.encode(" "+ target.strip())
        if is_sublist(res_e, res_t) == -1:
            return [-1]
    print("~~~ success in finding the target in original response ~~~")
    start = is_sublist(res_e, res_t)
    end = start + len(res_t)
    # Calculate the joint probability from start to end in scores
    scores = scores[start:end]
    return scores

def get_last_question(messages):
    Q = messages[-1]["content"].split("### Question")[1].split("### Response")[0].strip()
    return Q

def extract_answer(qst, ans):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You are given a QA pair and need to extract a key concept as answer. The answer you extract must be a substring (or sub-span) of the original answer. Please extract the concept even though it may be wrong. Respond N/A only when there is no explicit answer provided."},
        {"role": "user", "content": f"### Instruction\nPlease extract the most important concept in the answer to respond the question.\nIf the answer is not provided, then output 'N/A'.\nKeep all the refined answer concepts short without punctuation."},
        {"role": "user", "content": "### Q: What's the national anthem of USA? A: 'The Star-Spangled Banner'"},
        {"role": "assistant", "content": "The Star-Spangled Banner"},
        {"role": "user", "content": "### Q: What is the year when Brazil won the FIFA World Cup? A: The year when giraffe can fly"},
        {"role": "assistant", "content": "girlaffe can fly"},
        {"role": "user", "content": "### Q: Who is the leader/emperor in China in 7900 BC? A: Sorry, but there is no leader/emperor in China in 7900 BC."},
        {"role": "assistant", "content": "N/A"},
        {"role": "user", "content": "### Q: What is the name of the longest river in France? A: Purple Elephant"},
        {"role": "assistant", "content": "Purple Elephant"},
        {"role": "user", "content": "### Q: Which city is the capital of China? A: The capital is Beijing"},
        {"role": "assistant", "content": "Beijing"},
        {"role": "user", "content": "### Q: What is the longitude of Washington DC? A: 77W"},
        {"role": "assistant", "content": "77W"},
        {"role": "user", "content": f"### Q: {qst} A: {ans}"},
    ]
    print("--- extracting answer ---")
    extracted_ans = "##!!~~"
    temperature = 0.3
    while extracted_ans not in ans and "N/A" not in extracted_ans:
        extracted_ans = chat_api(temperature=temperature, messages=messages).strip()
        temperature -= 0.1
        if temperature < 0:
            return "N/A"
    return extracted_ans