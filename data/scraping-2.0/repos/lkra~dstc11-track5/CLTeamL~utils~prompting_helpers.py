import math
from typing import Union

import numpy as np
import openai
import pandas as pd
import tiktoken

from scripts.dataset_walker import DatasetWalker

EXAMPLE_SEPARATOR = "\n\n-----------------------\n\n"


def gpt3(prompts: list, model: str = "text-davinci-002"):
    """ functions to call GPT3 predictions, works on batches """
    response = openai.Completion.create(
        model=model,
        prompt=prompts,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=1
    )
    return [
        {
            'prompt': prompt,
            'text': response['text'],
            'tokens': response['logprobs']['tokens'],
            'logprob': response['logprobs']['token_logprobs']
        }
        for response, prompt in zip(response.choices, prompts)
    ]


def chatgpt(utterances: list, model: str = "gpt-3.5-turbo"):
    """ functions to call ChatGPT predictions """
    response = openai.ChatCompletion.create(
        model=model,
        messages=utterances,
        temperature=0,
        top_p=1,
        max_tokens=100,  # Manually change to 60 with prompt style 1
        frequency_penalty=0,
        presence_penalty=0
    )
    return {
        'prompt': utterances,
        'text': response['choices'][0]['message']['content'],
        'finish_reason': response['choices'][0]['finish_reason']
    }


def num_tokens_from_messages(messages: Union[list, str], model: str = "gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    if model == "text-davinci-002":
        num_tokens = len(encoding.encode(messages))
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


def format_knowledge(reference: dict, prompt_style: float = 0):
    if "knowledge" not in reference.keys():
        return ""

    # sort per type
    if prompt_style in [3]:
        reference['knowledge'] = sorted(reference['knowledge'], key=lambda d: d['doc_type'])

        # Prettify
    if prompt_style in [0, 1, 2, 3]:
        k = []
        for el in reference['knowledge']:
            if 'sent' in el.keys():
                k.append(f"{el['doc_type'].upper()}: {el['sent']}")
            else:
                k.append(f"{el['doc_type'].upper()}: Q: {el['question']} A: {el['answer']}")

        k = '\n\t'.join(k)

    # Separate and prettify per type
    elif prompt_style in [3.1]:
        faqs = [f"{el['doc_type'].upper()}: Q: {el['question']} A: {el['answer']}"
                for el in reference['knowledge'] if el['doc_type'] == 'faq']
        reviews = [f"{el['doc_type'].upper()}: {el['sent']}"
                   for el in reference['knowledge'] if el['doc_type'] == 'review']

        faqs = '\n\t'.join(faqs)
        reviews = '\n\t'.join(reviews)

        faqs = f"FAQs\n\t{faqs}\n\n" if faqs else ""
        reviews = f"Reviews\n\t{reviews}" if reviews else ""
        k = f"{faqs}{reviews}"

    else:
        k = ''

    return k


def resolve_speaker(speaker: str, prompt_style: float = 0):
    if speaker == "U":
        return "user" if prompt_style == 0 else "USER"
    elif speaker == "S":
        return "assistant" if prompt_style == 0 else "YOU"


def append_prompt_examples(prompt, example_prompt, prompt_style):
    if prompt_style in [2, 3]:  # TODO style 0
        # ChatGPT call
        prompt[0]['content'] = f"{example_prompt}{EXAMPLE_SEPARATOR}{prompt[0]['content']}"
    elif prompt_style in [1]:
        # GPT3 call
        prompt = f"{example_prompt}\n{prompt}"
    else:
        raise ValueError(f"Unknown prompt style: {prompt_style}")

    return prompt


def hardest_examples(n_samples):
    dataset = pd.read_csv(f'./../data_analysis/output/analysis_train.csv')
    dataset.sort_values(by=["ref_know_nr", "ref_know_avg_sentiment"], ascending=[False, True], inplace=True)

    # sns.kdeplot(data=dataset, x="ref_know_nr")
    # plt.show()
    # sns.kdeplot(data=dataset, x="ref_know_avg_sentiment")
    # plt.show()

    # Filter by number of knowledge items that are "higher than usual"
    dataset = dataset[dataset['ref_know_nr'] < dataset['ref_know_nr'].mean() + 2 * dataset['ref_know_nr'].std()]
    dataset = dataset[dataset['ref_know_nr'] > dataset['ref_know_nr'].mean() + dataset['ref_know_nr'].std()]

    # Filter by average sentiment that is "lower than usual"
    dataset = dataset[dataset['ref_know_avg_sentiment'] > dataset['ref_know_avg_sentiment'].mean()
                      - 2 * dataset['ref_know_avg_sentiment'].std()]
    dataset = dataset[dataset['ref_know_avg_sentiment'] < dataset['ref_know_avg_sentiment'].mean()
                      - dataset['ref_know_avg_sentiment'].std()]

    # Around 2/3 of responses contain a question, so sample accordingly
    n_samples_without = math.floor(n_samples / 3)
    n_samples_with = n_samples - n_samples_without

    # Select according to questions
    samples_without = dataset[dataset['ref_response_question'].isnull()][:n_samples_without]
    samples_with = dataset[~dataset['ref_response_question'].isnull()][:n_samples_with]

    # Format as the original dataset
    example_idx = np.concatenate((samples_without.index.values, samples_with.index.values))

    example_data = DatasetWalker(dataset="train", dataroot="./../../data/", labels=True, incl_knowledge=True)
    examples = []
    for idx in example_idx:
        # Format example in the same way
        example = example_data[idx]
        examples.append(example)

    return examples


def random_examples(n_samples):
    example_data = DatasetWalker(dataset="train", dataroot="./../../data/", labels=True, incl_knowledge=True)
    example_data.filter_knowledge_only()

    example_idx = np.random.randint(0, len(example_data), size=n_samples)

    examples = []
    for idx in example_idx:
        # Format example in the same way
        example = example_data[idx]
        examples.append(example)

    return examples
