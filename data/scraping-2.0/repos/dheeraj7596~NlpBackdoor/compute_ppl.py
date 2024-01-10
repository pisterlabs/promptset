import torch
import math
import pickle
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def ppl(sentence):
    global model, tokenizer
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    max_length = model.config.n_positions
    stride = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    lls = []
    count = 0
    for i in tqdm(range(1, tensor_input.size(1), stride)):
        begin_loc = max(i + stride - max_length - 1, 0)
        end_loc = i + stride - 1
        input_ids = tensor_input[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0]

        lls.append(log_likelihood)
        count += 1

    ppl = torch.exp(torch.stack(lls).sum() / count)
    return ppl.item()


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/NlpBackdoor/data/"
    basepath = "/data4/dheeraj/backdoor/"
    dataset = "imdb/"
    pkl_dump_dir = basepath + dataset

    df = pickle.load(open(pkl_dump_dir + "df_train_mixed_poisoned_clean.pkl", "rb"))
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    model = model.to(device)

    ppl_scores = []
    i = 0
    for sent in df.text:
        if i % 100 == 0:
            print("Number of sentences finished: ", i)
        ppl_scores.append(ppl(sent))
        i += 1
    pickle.dump(ppl_scores, open(pkl_dump_dir + "ppl_scores.pkl", "wb"))
