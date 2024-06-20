from nltk.translate import gleu_score
from tqdm import tqdm
from llm_async import run_llm_coroutine
import json
import os
import asyncio

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
        generated_correction = await run_llm_coroutine(prompt_interpolated, temperature=0.0, msg="Scoring - 30 calls mostly")
        assert len(generated_correction) == len(testing_sample)
        for i in range(len(generated_correction)):
            accuracy += gleu_score.sentence_gleu([generated_correction[i]], testing_sample[i]["correction"])
        prompt_score_pairs[prompt] = accuracy / len(testing_sample) * 100

    return prompt_score_pairs


if __name__ == "__main__":
    data = json.load(open("synthetic_dataset.json"))
    training_sample = data[:64]
    prompt = "Correct the grammar in the sentence: {TEXT}"
    print(asyncio.run(score([prompt], training_sample)))
    print(asyncio.run(score([prompt], training_sample)))
    print(asyncio.run(score([prompt], training_sample)))