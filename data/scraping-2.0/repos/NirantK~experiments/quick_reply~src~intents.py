import json
import random
from pathlib import Path

import openai
from sklearn.metrics import accuracy_score

keys_path = "keys.json"
openai.api_key = json.load(open(Path(keys_path).resolve()))["api_key"]


def intent_classification_one_feed_examples(
    intent_positive_examples: list, intent_negative_examples: list, intent: str
) -> str:
    """Changes the data in the required format for text classification
    This one is to classify_intents one intent at a time, i.e weather the given 
    sentence has the particular intent or not. 
    
    Ideally, use this when the examples/negative examples are a lot. 
    This is because GPT3 has constraints on max number of tokens in 
    prompt (2048)
    Args:
        intent_positive_examples (list): This is a list of all the positive
            example for the intent. 
        intent_negative_examples (list): This is a list of all the negative 
            examples for the intent
        intent (str): Name of the intent. Avoid writing verbs here and also 
            use in simple present. e.g use "buy" insted of "to buy"

    Returns:
        str: string that can be used for gpt3 prompt.
    """
    gpt3_class_input = f"Here we tell whether the intent is {intent} or not {intent}\n"
    pos_mix = ["S:" + sent + "\n" + f"I: {intent}" for sent in intent_positive_examples]
    neg_mix = [
        "S:" +sent + "\n" + f"I: not {intent}" for sent in intent_negative_examples
    ]
    mix = pos_mix + neg_mix
    random.shuffle(mix)
    gpt3_class_input = gpt3_class_input + "\n".join(mix)
    if len(gpt3_class_input.split()) > 950:
        print("The final input is exceeding the limit")
        return None
    return gpt3_class_input


def intent_classification_many_feed_examples(
    intents: list, intent_examples: dict
) -> str:
    """Changes the data in the required format for text classification.
    This can take multiple intents. Ideally use this when you have 2-3 
    examples per intent and total number of intents is considerably smaller.

    Args:
        intents (list): list of all the intents. e.g ["intent1", "intent2"]
        intent_examples (dict): Mapping intents to examples. Make sure the keys 
            here match the keys provided in the `intents` list
            {
                "intent1": [
                    "example 1 for intent 1",
                    "example 2 for intent 1 "
                ],
                "intent2":[
                    "example 1 for intent 2",
                    "example 2 for intent 2 "
                ]
            }
    Returns:
        str: string that can be used for gpt3 prompt.
    """
    gpt3_class_input = ""
    mix = []
    for intent in intents:
        mix += [f"S: {sent}\nI: {intent}" for sent in intent_examples[intent]]
    random.shuffle(mix)
    gpt3_class_input = gpt3_class_input + "\n".join(mix)
    if len(gpt3_class_input.split()) > 950:
        print(f"The final input is exceeding the limit. Len : {len(gpt3_class_input.split())}")
        return None
    return gpt3_class_input


def classify_intents(
    gpt3_class_input: str, query: str, return_entire_resp: str = False
) -> str:
    """

    Args:
        gpt3_class_input (str) : string to resemble the gpt3 classification 
            prompt. Use intent_classification_many_feed_examples or 
            intent_classification_one_feed_examples for intents
        query (str) : query
        return_entire_resp (str, optional): Set true if you want to return the entire output
            with logprobs. Defaults to False.

    Returns: return intent with format Intent : intentname or 
        entire json response if return_entire_resp is true
    """
    resp = openai.Completion.create(
        engine="davinci",
        prompt=gpt3_class_input + "\nS:" + query + "\n",
        max_tokens=10,
        temperature=0,
        logprobs=10,
    )
    if return_entire_resp:
        return resp 
    o = resp.choices[0].text
    return o[o.find("I:")+3 :o.find("\n")]


def evaluate_intents(in_: str, x: list, y: list) -> list:
    """Evaluates predicted intents

    Args:
        in_ (str): Prompt for classification
        x (list): list of queries of len n 
        y (list): intents of len n

    Returns:
        list: predicted intents of len n
    """
    y_pred = [classify_intents(in_, query).split()[1] for query in x]
    print(f"Accuracy Score : {accuracy_score(y, y_pred)}")
    return y_pred
