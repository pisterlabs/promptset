from llm_async import run_llm_coroutine
from tqdm import tqdm
from nltk.translate import gleu_score
from datasets import Dataset
import pandas as pd
import asyncio
import json
from sentence_transformers import SentenceTransformer, util

# Load transformer model
transformer_model = SentenceTransformer("all-mpnet-base-v2")  # Load transformer model
def similarity(text1, text2):
        embeddings = transformer_model.encode([text1, text2])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

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
        summaries_generated = await run_llm_coroutine(prompt_interpolated, temperature=0.0)
        assert len(summaries_generated) == len(testing_sample)
        for i in range(len(summaries_generated)):
            accuracy += similarity(testing_sample[i]["summary"], summaries_generated[i])
        prompt_score_pairs[prompt] = accuracy / len(testing_sample) * 100

    return prompt_score_pairs


if __name__ == "__main__":
    # Load the dataset
    true_dataset = pd.read_csv("summ_dataset.csv")
    # Select relevant columns
    true_dataset = true_dataset[["article", "highlights"]]
    # Rename the columns
    true_dataset.columns = ["text", "summary"]
    true_dataset = true_dataset.sample(70, random_state=42)
    
    print(true_dataset.head())

    # Loading optimized prompt
    initial_prompts = ["""Summarize the following text. Keep the original language in 
which the text is written. The summary has to be shorter than the original text. Don't add up or make up
any information not present in the original text.
Text: {TEXT}"""]
    optimized_prompts = ["To craft a paramount and supremely concise summary, thoroughly internalize the original text {TEXT}, meticulously discerning its intrinsic essence, nuanced subtleties, and underlying complexity. Ensure your summary harmoniously reflects the original text, scrupulously excluding any extraneous information not present, and maintaining a seamless, polished flow that echoes the original content's level of clarity, nuance, sophistication, and depth. Initialize your summary with 'In summary, ' and focus on preserving the core message, essential information, and underlying essence of the original text, thereby creating a distilled, yet comprehensive, overview that faithfully represents the original content, free from bias and stereotypes, while conveying the same level of insight, complexity, and precision, and ultimately, providing a masterful representation of the original text's underlying structure and logical flow."]

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
    
    df.to_csv("score_summ.csv", index=False)
