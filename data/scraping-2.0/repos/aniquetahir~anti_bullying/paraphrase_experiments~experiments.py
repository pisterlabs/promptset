from common_generative import get_summary_batched, ask_chatgpt,\
    extract_response, get_paraphrase_batched, get_response_batched, \
    extract_responses_paraphrase, PARAPHRASE_PROMPT_1 as PARAPHRASE_PROMPT_EXEMPLARS
from common import get_instagram_sessions
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle
import fire
from tqdm import tqdm
import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

PARAPHRASE_INSTRUCTION = """### Instruction:
Paraphrase the following text:

### Input:
{input}

### Response:
"""

SAMPLE_SIZE = 200
BATCH_SIZE = 30
PATH_LLAMA_7B = "/scratch/artahir/llama/hf_weights/llama2-7B"
PATH_LLAMA_13B = "/scratch/artahir/llama/hf_weights/llama2-13B"



def get_paraphrase_sample(sample_size=200, batch_size=7, exemplars=True, model=None, tokenizer=None):
    sessions = get_instagram_sessions()
    comment_idxs = []
    for i_session, session in enumerate(sessions):
        for i, c in enumerate(session['comments']):
            if len(c['content'].split(' ')) > 5:
                comment_idxs.append((i_session, i))

    # select a random sample of comment idxs
    chosen_sample_idxs = np.random.choice(len(comment_idxs), sample_size, replace=False)
    comment_idxs = [comment_idxs[i] for i in chosen_sample_idxs]

    if exemplars:
        # get the paraphrases for each comment
        paraphrases = []
        for i in tqdm(range(0, len(comment_idxs), batch_size)):
            # get the batch of comments
            comments = [sessions[s_i]['comments'][c_i]['content'] for s_i, c_i in comment_idxs[i:i+batch_size]]
            paraphrases.extend(zip(comments, get_paraphrase_batched(comments, model=model, tokenizer=tokenizer)))
        return paraphrases
    else:
        # get the paraphrases for each comment
        paraphrases = []
        for i in tqdm(range(0, len(comment_idxs), batch_size)):
            # get the batch of comments
            comments = [sessions[s_i]['comments'][c_i]['content'] for s_i, c_i in comment_idxs[i:i+batch_size]]
            prompts = [PARAPHRASE_INSTRUCTION.format(input=c) for c in comments]
            responses = get_response_batched(prompts, model=model, tokenizer=tokenizer)
            responses = [extract_response(r) for r in responses]
            paraphrases.extend(zip(comments, responses))
        return paraphrases


def get_paraphrase_sample_openai(sample_size=200, exemplars=True):
    sessions = get_instagram_sessions()
    comment_idxs = []
    for i_session, session in enumerate(sessions):
        for i, c in enumerate(session['comments']):
            if len(c['content'].split(' ')) > 5:
                comment_idxs.append((i_session, i))

    # select a random sample of comment idxs
    chosen_sample_idxs = np.random.choice(len(comment_idxs), sample_size, replace=False)
    comment_idxs = [comment_idxs[i] for i in chosen_sample_idxs]

    if exemplars:
        # get the paraphrases for each comment
        paraphrases = []
        for i in tqdm(range(0, len(comment_idxs))):
            s_i, c_i = comment_idxs[i]
            comment = sessions[s_i]['comments'][c_i]['content']
            # create the prompt based on the comment
            try:
                prompt = PARAPHRASE_PROMPT_EXEMPLARS.format(text=comment)
                paraphrases.append((comment, ask_chatgpt(prompt)))
            except Exception as e:
                print(e)
            # get the batch of comment
        return paraphrases
    else:
        # get the paraphrases for each comment
        paraphrases = []
        for i in tqdm(range(0, len(comment_idxs))):
            s_i, c_i = comment_idxs[i]
            comment = sessions[s_i]['comments'][c_i]['content']
            # create the prompt based on the comment
            try:
                prompt = PARAPHRASE_INSTRUCTION.format(input=comment)
                paraphrases.append((comment, ask_chatgpt(prompt)))
            except Exception as e:
                print(e)
            # get the batch of comments
        return paraphrases

def paraphrase_experiment_local_model(model_path: str, exemplars=True, sample_size=SAMPLE_SIZE, batch_size=BATCH_SIZE, pad_is_eos=True):
    model = AutoModelForCausalLM.from_pretrained(
         model_path,
         load_in_8bit=True, device_map='auto')
    # use better transformer
    model.to_bettertransformer()
    model.tie_weights()
    # compile model
    model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding=True, padding_side='left')
    if pad_is_eos:
        tokenizer.pad_token = tokenizer.eos_token
    return get_paraphrase_sample(sample_size=sample_size, exemplars=exemplars, batch_size=batch_size, model=model, tokenizer=tokenizer)

def paraphrase_experiment_llama7B(exemplars=True):
    paraphrases = paraphrase_experiment_local_model(PATH_LLAMA_7B, exemplars=exemplars)
    eos_token = "</s>"
    paraphrases = [(c, r.replace(eos_token, '')) for c, r in paraphrases]
    return paraphrases

def paraphrase_experiment_llama13B(exemplars=True):
    paraphrases = paraphrase_experiment_local_model(PATH_LLAMA_13B, exemplars=exemplars)
    eos_token = "</s>"
    paraphrases = [(c, r.replace(eos_token, '')) for c, r in paraphrases]
    return paraphrases

def paraphrase_experiment_gpt2(exemplars=True):
    paraphrases = paraphrase_experiment_local_model('gpt2-medium', exemplars=exemplars)
    eos_token = "<|endoftext|>"
    # remove eos token
    paraphrases = [(c, r.replace(eos_token, '')) for c, r in paraphrases]
    return paraphrases

def paraphrase_experiment_chatgpt(exemplars=True):
    return get_paraphrase_sample_openai(exemplars=exemplars)

arg_to_func = {
    '7b': paraphrase_experiment_llama7B,
    '13b': paraphrase_experiment_llama13B,
    'gpt2': paraphrase_experiment_gpt2,
    'chatgpt': paraphrase_experiment_chatgpt
}

def main(model:str, exemplars:bool=True):
    paraphrases = arg_to_func[model](exemplars)
    exemplar_str= 'exemplars' if exemplars else 'noexemplars'
    with open(f'paraphrases_{model}_{exemplar_str}.pkl', 'wb') as f:
        pickle.dump(paraphrases, f)

if __name__ == "__main__":
    fire.Fire(main)


