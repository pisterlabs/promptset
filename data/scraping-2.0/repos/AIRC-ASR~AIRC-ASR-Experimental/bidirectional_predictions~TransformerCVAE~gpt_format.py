import openai
import json
import os
import traceback
import torch.nn as nn
from transformers import GPT2Tokenizer
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

from data.util import prepare_dataset

num_stories = 200


def load_plots(num_stories: int):
    with open("bidirectional_predictions/TransformerCVAE/data/wikiPlots/titles", "r") as file:
        titles = file.read().split("\n")[:num_stories]

    with open("bidirectional_predictions/TransformerCVAE/data/wikiPlots/plots_paragraph", "r") as file:
        stories = []
        for i in range(num_stories):
            story = ""
            line = file.readline()
            while line[:5] != "<EOS>":
                story += line
                line = file.readline()
            stories.append(story)
            story = ""
    return titles, stories


def prompt_to_story(titles: list[str], stories: list[str]):
    examples = []

    for i in range(len(titles)):
        examples.append({"prompt": titles[i], "completion": f" {stories[i]}"})

    return examples


def sentence_to_sentence(titles: list[str], stories: list[str], prompt_len=1, predict_len=1):
    examples = []

    for story in stories:
        story = story.split(".")
        if len(story[-1]) < 3:
            story.pop()
        while len(story) > prompt_len+predict_len:
            prompt = ""
            size = 0
            while size < prompt_len:
                prompt += story.pop(0)+"."
                size += 1

            predict = ""
            size = 0
            while size < predict_len:
                predict += story.pop(0)+"."
                size += 1

            examples.append({"prompt": prompt, "completion": f" {predict}"})
            
    return examples


def save_examples(examples: list[dict[str, str]], save_file: str):
    with open(save_file, "w") as file:
        for example in examples:

            file.write(json.dumps(example)+"\n")


def upload_file(filename: str, friendly_name: str=None):
    return openai.File.create(file=open(filename, "rb"), purpose='fine-tune', user_provided_filename=friendly_name)


def create_fine_tune(training_file: str, base_model: str):
    return openai.FineTune.create(training_file=training_file, model=base_model)


def eval_model(model_name: str):

    print('Setup data...')
    seq_len = 1024
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='out/model_cache')
    tokenizer.max_len = int(1e12)
    test_loader = prepare_dataset("data", "wi", tokenizer, 1, seq_len, 1, seq_len, 1, seq_len,
        make_train=False, make_val=False, make_test=True, data_type="t1")[0]

    print('Done.')

    loss_fn = nn.CrossEntropyLoss(reduction='none')


    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    startofcond = tokenizer.convert_tokens_to_ids("<|startofcond|>")
    endofcond = tokenizer.convert_tokens_to_ids("<|endofcond|>")

    n_samples = 0
    bleu4_sum = 0.0
    rouge_scores_values_sum = [0.0] * 9

    model_type = "cvae"

    # test_iter = iter(test_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(test_iter)
    with tqdm(total=len(test_loader)) as pbar:
        for i_test, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(test_loader):

            eff_samples = []
            n, l = target_tokens.size()
            storys = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
            storys_str = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in storys]

            response = openai.Completion.create(model=model_name, prompt="Say this is a test", max_tokens=7, temperature=0)
            text = [response["choices"][0]["text"]]

            # score for one long text, higher than 0.075 usually means repetition
            # rep_score = repeat_score(text.split(), ngram=[3, 4, 5, 6, 7, 8])
            # if rep_score > 0.075:
            #     # print(rep_score)
            #     continue

            try:
                # check bleu
                bleu4 = sentence_bleu([storys_str[0].split()], [text[0].split()], smoothing_function=SmoothingFunction().method7)

                # check rouge
                rouge = Rouge()
                rouge_scores = rouge.get_scores(text[0], storys_str[0])

                bleu4_sum += bleu4
                n_samples += 1
            except Exception as e:
                print(traceback.format_exception(e))
                bleu4 = 0.0
                rouge_scores = [{'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                 'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                 'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]

            rouge_scores_values = [v for k in rouge_scores[0].keys() for v in rouge_scores[0][k].values()]
            rouge_scores_values_sum = [v1 + v2 for v1, v2 in zip(rouge_scores_values_sum, rouge_scores_values)]

            eff_samples.append((text, bleu4, rouge_scores))

            pbar.update(1)

            print(' bleu-4:', bleu4)
            print(' rouge :', rouge_scores_values)

    print('Test complete with %05d samples.' % n_samples)

    bleu4 = round(bleu4_sum / n_samples, 3)
    rouge_scores_values = [round(r / n_samples, 3) for r in rouge_scores_values_sum]
    print(' bleu-4:', bleu4)
    print(' rouge :', rouge_scores_values)


if __name__ == "__main__":
    openai.api_key = open("../../creds/OPENAI_API_KEY", "r").read()
    # titles, stories = load_plots(num_stories)
    # eval_model("text-davinci-003")
    # print(upload_file("gpt_data.jsonl"))
    # ft-DbQ3URrMSQNxdAFODc3u7D4p
    # file-8rrm3nBuF8mcfYFUQfKF6Q1b
    # ft-bo1UZxw9qReOXWU8N1iUAWpA
    # file-n7xbkIjE3CcXJjP5Ixle0DRQ
    # ft-Xc0xrj8cZMpIDoFh7jw2dx1T
    # print(create_fine_tune("file-n7xbkIjE3CcXJjP5Ixle0DRQ", "davinci"))
    print(openai.FineTune.list())
    # print(openai.FineTune.retrieve(id="ft-DbQ3URrMSQNxdAFODc3u7D4p"))
    # examples = sentence_to_sentence(titles, stories, 4, 2)
    # examples = prompt_to_story(titles, stories)
    # save_examples(examples, "bidirectional_predictions/TransformerCVAE/data/gpt_data.jsonl")
    print("edn")
