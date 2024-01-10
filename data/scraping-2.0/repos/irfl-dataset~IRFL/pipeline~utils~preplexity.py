import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel


def model_init(model_string, cuda):
    if model_string.startswith("gpt2"):
        tokenizer = GPT2Tokenizer.from_pretrained(model_string)
        model = GPT2LMHeadModel.from_pretrained(model_string)
    else:
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_string)
        model = OpenAIGPTLMHeadModel.from_pretrained(model_string)
    model.eval()
    if cuda:
        model.to('cuda')
    print("Model init")
    return model, tokenizer


def sent_scoring(model_tokenizer, text, cuda):
    model = model_tokenizer[0]
    tokenizer = model_tokenizer[1]
    try:
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
        if cuda:
            input_ids = input_ids.to('cuda')
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        sentence_prob = loss.item()
        return sentence_prob
    except:
        return -1

model, tokenizer = model_init('gpt2', False)

def get_perplexity(sentence):
    return sent_scoring((model, tokenizer), sentence, False)/len(sentence)

