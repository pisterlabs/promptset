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
    true_dataset = pd.read_csv("Grammar Correction.csv")
    # true_dataset = true_dataset[true_dataset["Error Type"] == "Punctuation Errors"][["Ungrammatical Statement", "Standard English"]]
    true_dataset = true_dataset[["Ungrammatical Statement", "Standard English"]]
    # Rename the columns
    true_dataset.columns = ["text", "correction"]
    true_dataset = true_dataset.sample(70, random_state=42)

    # Loading optimized prompts
    prompt_scores = json.load(open("testingSetScores.json"))
    initial_prompts = []
    optimized_prompts = []
    initial_scores = []
    optimized_scores = []

    for prompt, s in list(prompt_scores.items()):
        data = prompt_scores[prompt]
        initial_prompts.append(list(data["initial_prompt"].keys())[0])
        optimized_prompts.append(list(data["optimized_prompt"].keys())[0])
        initial_scores.append(list(data["initial_prompt"].values())[0])
        optimized_scores.append(list(data["optimized_prompt"].values())[0])


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
        "initial_synth_score": initial_scores,
        "optimized_synth_score": optimized_scores,
        "initial_true_score": initial_scores_trueDataset,
        "optimized_true_score": optimized_scores_trueDataset
    })
    
    df.to_csv("scores_errCorr.csv", index=False)
