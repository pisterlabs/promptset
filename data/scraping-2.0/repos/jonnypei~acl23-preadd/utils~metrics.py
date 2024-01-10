import json
import numpy as np
import openai
import torch
import torch.nn as nn
import torch.nn.functional as F

from googleapiclient import discovery
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification
from .constants import OPENAI_API_KEY, PERSPECTIVE_API_KEY, PERSPECTIVE_API_ATTRIBUTES, EOT_TOKEN
from .utils import unpack_scores

openai.api_key = OPENAI_API_KEY

def get_perspective_api_scores(content, display=False):
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=PERSPECTIVE_API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        'comment': {'text': content},
        'requestedAttributes': dict([(attribute, {}) for attribute in PERSPECTIVE_API_ATTRIBUTES]),
        'languages': ["en"]
    }

    response = client.comments().analyze(body=analyze_request).execute()

    summary_scores, span_scores = unpack_scores(response)

    return summary_scores, span_scores


def perplexity(sentences, device='cuda'):
    ppl_tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
    ppl_model = AutoModelWithLMHead.from_pretrained('openai-gpt').to(device)
    ppl_model.eval()

    # calculate perplexity
    with torch.no_grad():
        ppl = []
        sos_token = ppl_tokenizer.decode([0])
        for sentence in tqdm(sentences, total=len(sentences)):
            full_tensor_input = ppl_tokenizer.encode(
                sos_token + sentence.replace(EOT_TOKEN, ' ').strip(), return_tensors='pt').to(device)[:512]
            full_loss = ppl_model(full_tensor_input, labels=full_tensor_input)[
                0].mean()
            ppl.append(torch.exp(full_loss).flatten().cpu().item())
    return ppl, np.mean(ppl), np.std(ppl)


def grammaticality(sentences, device='cuda'):
    gram_tokenizer = AutoTokenizer.from_pretrained(
        'textattack/roberta-base-CoLA')
    gram_model = AutoModelForSequenceClassification.from_pretrained(
        'textattack/roberta-base-CoLA').to(device)
    gram_model.eval()

    # calculate grammaticality
    with torch.no_grad():
        good_probs = []
        for sentence in tqdm(sentences, total=len(sentences)):
            good_prob = F.softmax(gram_model(gram_tokenizer.encode(
                sentence, return_tensors='pt').to(device))[0].flatten(), dim=0)[1]
            good_probs.append(good_prob.cpu().item())
    return good_probs, np.mean(good_probs), np.std(good_probs)

def fluency(prompt, generated_text):
    response = openai.Completion.create(
    engine='davinci',
    prompt=prompt,
    max_tokens=0,
    temperature=0.0,
    logprobs=0,
    echo=True,
    )
    prompt_logprobs = response['choices'][0]['logprobs']['token_logprobs'][1:]

    response = openai.Completion.create(
        engine='davinci',
        prompt=generated_text,
        max_tokens=0,
        temperature=0.0,
        logprobs=0,
        echo=True,
    )
    logprobs = response['choices'][0]['logprobs']['token_logprobs'][1:]

    continuation_logprobs = logprobs[len(prompt_logprobs):]
    return np.exp(-np.mean(continuation_logprobs))