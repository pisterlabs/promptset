import os, sys
from datasets import load_dataset
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
import openai
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
from dotenv import load_dotenv
import rag_uncertainty
import math
from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration, AutoModelForSequenceClassification
import torch
import json


def add_home_directory_to_path():
    home_dir = os.path.expanduser("~")
    if home_dir not in sys.path:
        sys.path.append(home_dir)

def answer_logprob(choices):
    logprobs = choices["logprobs"]["token_logprobs"]
    logprob = np.sum(logprobs)
    return logprob

def filter_unique_items(answers, logprobs):
    if len(answers) != len(logprobs):
        raise ValueError("Both lists must have the same length")

    item_occurrences = {}
    unique_items = []
    unique_labels = []
    repeat_times = {}

    for item, label in zip(answers, logprobs):
        if item in item_occurrences:
            item_occurrences[item] += 1
        else:
            item_occurrences[item] = 1
            unique_items.append(item)
            unique_labels.append(label)

    for item, count in item_occurrences.items():
        repeat_times[item] = count

    return unique_items, unique_labels, repeat_times


"""
Adopt from semantic uncertainty paper
"""
def get_predictive_entropy_over_concepts(semantic_logprob, semantic_set_ids, repeat_times):
    """Compute the semantic entropy"""

    # cluster logprobs by semantic meaning
    sets = {}
    for key in semantic_set_ids.keys():
        set = semantic_set_ids[key]
        if set not in sets:
            # sets[set] = [[semantic_logprob[key], repeat_times[key]]]
            sets[set] = [math.exp(semantic_logprob[key]) * repeat_times[key]]
        else:
            sets[set].append(math.exp(semantic_logprob[key]) * repeat_times[key])
    
    # compute p(c \mid x) for each concept c
    concept_probs = []
    totoal_prob = 0
    for set in sets.keys():
        # get logprobs for each concept c
        probs = torch.tensor(sets[set])
        # compute \sum_s p(s \mid x)
        concept_prob = torch.sum(probs)
        totoal_prob += concept_prob
        concept_probs.append(concept_prob)
    return [concept_prob/totoal_prob for concept_prob in concept_probs]
    

def compute_semantic_similarity(model, tokenizer, question, answers, logprobs):
    # filter out non-unique answers
    unique_generated_texts, logprobs, repeat_times = filter_unique_items(answers, logprobs)

    # unique_generated_texts = list(set(answers))
    semantic_set_ids = {}
    semantic_logprob = {}
    for index, answer in enumerate(unique_generated_texts):
        semantic_set_ids[answer] = index
    
    print('Number of unique answers:', len(unique_generated_texts))

    with torch.no_grad():
        if len(unique_generated_texts) > 1:

            # Evalauate semantic similarity
            for i, reference_answer in enumerate(unique_generated_texts):
                semantic_logprob[unique_generated_texts[i]] = logprobs[i]
                for j in range(i + 1, len(unique_generated_texts)):

                    qa_1 = question + ' ' + unique_generated_texts[i]
                    qa_2 = question + ' ' + unique_generated_texts[j]

                    input = qa_1 + ' [SEP] ' + qa_2
                    encoded_input = tokenizer.encode(input, padding=True)
                    prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
                    predicted_label = torch.argmax(prediction, dim=1)

                    reverse_input = qa_2 + ' [SEP] ' + qa_1
                    encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
                    reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
                    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

                    # print(qa_1, qa_2, predicted_label, reverse_predicted_label)
                    if 0 in predicted_label or 0 in reverse_predicted_label:
                        has_semantically_different_answers = True
                        deberta_prediction = 0
                    else:
                        semantic_set_ids[unique_generated_texts[j]] = semantic_set_ids[unique_generated_texts[i]]
    concept_probs = get_predictive_entropy_over_concepts(semantic_logprob, 
                                                     semantic_set_ids,
                                                     repeat_times)
    return concept_probs

def ask_chatgpt(question, context, semantic_model, semantic_tokenizer, few_shot):

    prompt = f"{question} {context}"
    prompt = few_shot + '\n\n' + prompt

    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=50,
        n=5,
        stop=None,
        temperature=0.5,
        logprobs=5
    )
    choices = [choice.text.strip() for choice in response.choices]
    logprobs = [answer_logprob(choice) for choice in response.choices]
    probs = [math.exp(logprob) for logprob in logprobs]

    concept_probs = compute_semantic_similarity(model=semantic_model, 
                                          tokenizer=semantic_tokenizer, 
                                          question=prompt, answers=choices, 
                                          logprobs=logprobs)

    return choices, probs, concept_probs


def load_natural_questions_dataset(split):
    breakpoint()
    dataset = load_dataset(
        "natural_questions",
        split=split,
        beam_runner="DirectRunner",
        beam_options=PipelineOptions(
            [
                "--direct_num_workers=4",
                "--direct_running_mode=multi_threading",
            ]
        ),
        cache_dir="/data3/shuoli/data/",
        ignore_verifications=True
    )
    return dataset

# def load_natural_questions_dataset(split="train"):
#     dataset = load_dataset("natural_questions", split=split, beam_runner='DirectRunner', cache_dir="/data3/shuoli/data/")
#     return dataset

def load_hotpotqa_dataset(split, config='fullwiki'):
    dataset = load_dataset("hotpot_qa", config, split=split, cache_dir="/data3/shuoli/data/")
    return dataset

def load_triviaQA_dataset(split="train"):
    """
    Load the TriviaQA dataset from Hugging Face Datasets.

    Args:
        split (str, optional): The dataset split to load. Options are 'train', 'validation', and 'test'. Defaults to 'train'.

    Returns:
        Dataset: The TriviaQA dataset split.
    """
    dataset = load_dataset("trivia_qa", "rc", split=split, cache_dir="/data3/shuoli/data/")
    return dataset

def load_FEVER_dataset(split="train"):
    """
    Load the FEVER dataset from Hugging Face Datasets.

    Args:
        split (str, optional): The dataset split to load. Options are 'train', 'validation', and 'test'. Defaults to 'train'.

    Returns:
        Dataset: The FEVER dataset split.
    """
    dataset = load_dataset("fever", 'v2.0', split=split, cache_dir="/data3/shuoli/data/")
    return dataset

def main():
    # Set up chatgpt api
    dotenv_path = Path('.env')
    load_dotenv(dotenv_path=dotenv_path)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.Model.list()

    # Choose the desired dataset split, e.g., 'train', 'validation', or 'test'
    split = "validation"
    # qa_dataset = load_hotpotqa_dataset(split)
    # qa_dataset = load_natural_questions_dataset(split)
    # qa_dataset = load_triviaQA_dataset(split)
    qa_dataset = load_FEVER_dataset(split=split)
    print(f"Loaded {split} dataset with {len(qa_dataset)} samples.")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], 
                                      use_stemmer=True)
    rouge_scores = []

    # Load the RAG model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
    # retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", 
    #                                          index_name="exact",
    #                                          use_dummy_dataset=True,
    #                                          n_docs=5)
    # model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", 
    #                                               use_dummy_dataset=True)

    """
    dummy index
    """
    # tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
    # retriever = RagRetriever.from_pretrained(
    #     "facebook/rag-token-nq", index_name="compressed", n_docs=5, use_dummy_dataset=True
    # )
    # # initialize with RagRetriever to do everything in one forward call
    # model = RagTokenForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever).cuda()

    # NLI model and tokenizer
    semantic_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    semantic_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()

    # read in file "NQ_fewshot.txt"
    with open('FEVER_fewshot.txt', 'r') as f:
        content = f.read()
        # print(content)
    
    predicted_probs = []
    # Process questions from the dataset
    for idx, sample in enumerate(qa_dataset):
        breakpoint()
        question = sample["question"]
        # context = sample["context"]
        context = sample["entity_pages"]["wiki_context"][0]
        reference_answer = sample["answer"]
        # question = sample["question"]
        # context = ["context"]
        
        true_context_scores = []

        # docs_dict, doc_scores, all_docs = \
        #     rag_uncertainty.retrieve(model, tokenizer, 
        #                              retriever, question, n_docs=5)
        # docs_titles = all_docs['title']
        # docs_texts = all_docs['text']

        # Send the question to the ChatGPT API

        ## 1 use true context
        predicted_answers, predicted_logprobs, concept_probs \
            = ask_chatgpt(question, context, 
                          semantic_model=semantic_model, 
                          semantic_tokenizer=semantic_tokenizer)

        predicted_probs.append(predicted_logprobs)
        # Calculate ROUGE scores
        for predicted_answer in predicted_answers:
            scores = scorer.score(reference_answer, predicted_answer)
            true_context_scores.append(scores)

        ## 2 use retrieved context
        retrieved_context_scores = []
        for text in docs_texts:
            predicted_answers, predicted_logprobs, concept_probs \
                = ask_chatgpt(question, text, 
                              semantic_model=semantic_model, 
                              semantic_tokenizer=semantic_tokenizer)

            predicted_probs.append(predicted_logprobs)
            # Calculate ROUGE scores
            for predicted_answer in predicted_answers:
                scores = scorer.score(reference_answer, predicted_answer)
                retrieved_context_scores.append(scores)

        # Stop after processing 5 samples to keep the output short
        if idx >= 4:
            break
    
    with open('predicted_probs.json', "w") as f:
        json.dump(predicted_probs, f)
    with open('rouge_scores.json', "w") as f:
        json.dump(rouge_scores, f)


    # Print the ROUGE scores for the processed samples
    # for idx, scores in enumerate(rouge_scores):
    #     print(f"Sample {idx + 1}:")
    #     for rouge_type, score in scores.items():
    #         print(f"{rouge_type}: {score.fmeasure:.4f}")

if __name__ == "__main__":
    main()
