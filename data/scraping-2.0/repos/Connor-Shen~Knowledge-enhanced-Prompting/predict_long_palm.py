from Prompts_palm import meta_prompt, scorer_prompt, prompts
import pandas as pd
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.llms.google_palm import GooglePalm
from langchain.callbacks.base import BaseCallbackHandler
import re
import os

def create_chain_from_template(template, input_variables, temperature=0.5, callbacks=None, verbose=True):
    """
    Create a language model chain from a template.
    Args:
        template (str): The template for generating prompts.
        input_variables (list): List of input variables used in the template.
        temperature (float): The temperature parameter for the language model.
        callbacks (list): List of callback handlers.
        verbose (bool): Whether to print verbose output.
    Returns:
        LLMChain: The language model chain.
    """
    prompt = PromptTemplate(
        input_variables=input_variables,
        template=template
)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    # default "text-bison-001"
    chain = LLMChain(
        llm= GooglePalm(google_api_key=google_api_key, temperature = temperature),
        prompt=prompt,
        callbacks=callbacks if callbacks is not None else [],
        verbose=verbose,
    )

    return chain

# calculate accuracy, recall, precision, and fb score
def calculate_metrics(predictions, labels):

    TP = sum([(p == 1) and (l == 1) for p, l in zip(predictions, labels)])
    TN = sum([(p == 0) and (l == 0) for p, l in zip(predictions, labels)])
    FP = sum([(p == 1) and (l == 0) for p, l in zip(predictions, labels)])
    FN = sum([(p == 0) and (l == 1) for p, l in zip(predictions, labels)])

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # calculate the weighted f1 score
    beta = 2   # emphasize recall
    fb = ((beta**2 + 1) * precision * recall) / ((beta**2 * precision) + recall) if (precision != 0 or recall != 0) else 0

    return accuracy, recall, precision, fb


# modify the score_prompts function
def score_evaluates(scorer_chain, prompts, training_examples, performance_df):

    labels = training_examples['label']
    for prompt in prompts:
        predictions = []
        for index, example in training_examples.iterrows():
            question = example['text']
            # in the form of 0 and 1 
            sample_answer = scorer_chain.predict(question=question, instruction=prompt)
            predictions.append(1 if sample_answer == '1' else 0)
        accuracy, recall, precision, fb_score = calculate_metrics(predictions, labels)
        performance_df = performance_df.append({
            'text': prompt,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'fb_score': fb_score,
        }, ignore_index=True)
        # save the checkpoint
        performance_df.to_excel("checkpoint/performance_log.xlsx", index=False)
    return performance_df

def predict_eval(scorer_chain, performance_df, training_examples, prompts):
    """
    return evaluation scores of different prompts
    Args:
        scorer_chain (LLMChain): Scorer language model chain.
        performance_df (pd.DataFrame): DataFrame containing text and scores.
        training_examples (pd.DataFrame): DataFrame containing training exemplars.
        prompts: list of prompts
    Returns:
        pd.DataFrame: Updated performance DataFrame.
    """
    performance_df = score_evaluates(scorer_chain, prompts, training_examples, performance_df)
    return performance_df


if __name__ == "__main__":
    scorer_chain = create_chain_from_template(scorer_prompt,
                                                ["question", "instruction"],
                                                temperature=0,
                                                verbose=True)
    
    performance_df = pd.read_excel("data/performance_predict.xlsx")
    training_data_df = pd.read_excel("data/training_data.xlsx")

    evaluation_df = predict_eval(scorer_chain, performance_df, training_data_df, prompts)
    print(evaluation_df)
    evaluation_df.to_excel("data/performance_predict_final.xlsx", index=False)