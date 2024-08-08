from llm_async import run_llm_coroutine
from tqdm import tqdm
from nltk.translate import gleu_score
from datasets import Dataset
import pandas as pd
import asyncio
import json
from nltk.translate.bleu_score import sentence_bleu

async def score(prompts, testing_sample):
    """
    Score the instruction using the sample.

    Args:
    instruction: str
    sample: Dataset with "question" and "answer" as keys

    Returns:
    accuracy: float
    """
    bleu_score = lambda expected, actual: sentence_bleu(
        [expected.split()], actual.split(), 
        weights=[1],
    )

    prompt_score_pairs = {}
    for prompt in tqdm(prompts, desc="Scoring"):
        accuracy = 0
        prompt_interpolated = [prompt.format(TEXT=data_pair["text"]) for data_pair in testing_sample]
        generated_translation = await run_llm_coroutine(prompt_interpolated, temperature=0.0)
        assert len(generated_translation) == len(testing_sample)
        for i in range(len(generated_translation)):
            accuracy += bleu_score(testing_sample[i]["translation"], generated_translation[i])
        prompt_score_pairs[prompt] = accuracy / len(testing_sample) * 100

    return prompt_score_pairs


if __name__ == "__main__":
    # Load the dataset
    true_dataset = pd.read_csv("en-sp.csv")
    # Rename the columns
    true_dataset.columns = ["text", "translation"]
    true_dataset = true_dataset.sample(70, random_state=42)
    
    print(true_dataset.head())

    # Loading optimized prompt
    initial_prompts = ["""Please help me to translate the following text. Please return only translated content not include the origin text. Here is the text: 

{TEXT}"""]
    optimized_prompts = ["Think step by step to translate the text accurately. You are an expert translator, and your output will be evaluated by a panel of linguists. You will be penalized for any inaccuracies or inconsistencies in your translation. Please use the same language and tone as the original text to ensure clarity and coherence. Refer to the following example translations to guide your approach: [insert examples]. Here is the text to be translated: {TEXT}. Please return only the translated content, excluding the original text."]

    # Convert to Dataset
    true_dataset = Dataset.from_pandas(true_dataset)
    
    res = asyncio.run(score(initial_prompts, true_dataset))
    assert list(res.keys()) == initial_prompts
    initial_scores_trueDataset = list(res.values())
    
    res = asyncio.run(score(optimized_prompts, true_dataset))
    assert list(res.keys()) == optimized_prompts
    optimized_scores_trueDataset = list(res.values())
    
    df = pd.DataFrame({
        "initial_prompt": initial_prompts,
        "optimized_prompt": optimized_prompts,
        "initial_true_score": initial_scores_trueDataset,
        "optimized_true_score": optimized_scores_trueDataset
    })
    
    df.to_csv("score_translation.csv", index=False)
