from llm_async import run_llm_coroutine
from tqdm import tqdm
from nltk.translate import gleu_score
from datasets import Dataset
import pandas as pd
import asyncio
import json

async def score(prompts, testing_sample):
    """
    Score the instruction using the sample.

    Args:
    instruction: str
    sample: Dataset with "question" and "answer" as keys

    Returns:
    accuracy: float
    """
    prompt_score_pairs = {}
    for prompt in tqdm(prompts, desc="Scoring"):
        accuracy = 0
        prompt_interpolated = [prompt.format(TEXT=data_pair["text"]) for data_pair in testing_sample]
        generated_correction = await run_llm_coroutine(prompt_interpolated, temperature=0.0)
        assert len(generated_correction) == len(testing_sample)
        for i in range(len(generated_correction)):
            accuracy += gleu_score.sentence_gleu([generated_correction[i]], testing_sample[i]["correction"])
        prompt_score_pairs[prompt] = accuracy / len(testing_sample) * 100

    return prompt_score_pairs


if __name__ == "__main__":
    # Load the dataset
    true_dataset = pd.read_csv("all_errCorr_results/Grammar Correction.csv")
    # true_dataset = true_dataset[true_dataset["Error Type"] == "Punctuation Errors"][["Ungrammatical Statement", "Standard English"]]
    true_dataset = true_dataset[["Ungrammatical Statement", "Standard English"]]
    # Rename the columns
    true_dataset.columns = ["text", "correction"]
    true_dataset = true_dataset.sample(70, random_state=42)
    
    print(true_dataset.head())

    # Loading optimized prompt
    initial_prompts = ["Please properly punctuate the given text (without omitting a single word) and output only the resulting punctuated text. Please do not omit a single word from the original text. {TEXT}"]
    optimized_prompts = ["Your task is to meticulously refine the given text by masterfully inserting precise punctuation marks, thoughtfully preserving its original meaning, tone, and style without omitting or altering a single word. Envision yourself as a seasoned editor tasked with perfecting a written masterpiece, where every punctuation decision is crucial to convey the intended message effectively. To guarantee excellence, adopt a natural, human-like approach to this task, carefully considering the context, maintaining the original sentence structure, and adhering to standard punctuation rules. Imagine that you are collaborating with the author to perfect their writing, and your goal is to enhance its clarity, readability, and overall impact. Ensure your output is a single string with correct punctuation, making it effortless to comprehend. Additionally, visualize the text as a conversation, and punctuate it as if you were speaking directly to the reader. Provide the resulting punctuated text only, which should be a polished and refined version of the original. {TEXT}"]

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
    
    df.to_csv("score_errCorr.csv", index=False)
