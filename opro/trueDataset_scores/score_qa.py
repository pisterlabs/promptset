from llm_async import run_llm_coroutine
from tqdm import tqdm
from nltk.translate import gleu_score
from datasets import load_dataset, Dataset
import pandas as pd
import asyncio
import json
import re, os

async def create_scoring_prompt(prompt, sample_data):
    """
    Given a prompt and sample data, generates a scoring prompt for the prompt.
    """    
    prompt_template = f"""Write a scoring prompt for an example input output pair on a prompt to a language model. 
Use the variable name output for the output of the prompt. 

### Rules ###
- The scoring prompt must contain the "{{output}}" variable and "{{text}}" variable. Ensure that both variables are present in the scoring prompt criteria.
- Your answer should be inside <BEGIN_CRITERIA> and <END_CRITERIA> constructs.

## Example:
<BEGIN_PROMPT> 'what is a fruit of color: {{TEXT}}. Return the name of the fruit and nothing else:' <END_PROMPT>
<BEGIN_EXAMPLE_INPUT> {{"text": "yellow", "output": "banana"}} <END_EXAMPLE_INPUT>
<BEGIN_CRITERIA> Is a ${{output}} this color: ${{text}}? Answer yes or no only. <END_CRITERIA>

## Query:
<BEGIN_PROMPT> {prompt} <END_PROMPT>
<BEGIN_EXAMPLE_INPUT> {sample_data} <END_EXAMPLE_INPUT>"""

    # Extract the scoring prompt
    for i in range(10):
        try:
            # Generate Scoring Prompt
            res = await run_llm_coroutine([prompt_template], model="llama3-70b", temperature=1.0)
            res = res[0]

            # Extract Criteria
            match = re.search(r'<BEGIN_CRITERIA>(.*?)<END_CRITERIA>', res, re.DOTALL)
            assert match is not None, "No match found for <BEGIN_CRITERIA> and <END_CRITERIA> tags"
            extracted_text = match.group(1).strip()
            
            # Check if extracted text has the correct keywords
            matches = re.findall(r"{[^}]*}", extracted_text)
            assert matches is not None, "No matches found for variables in prompt"
            assert len(matches) == 2, "Prompt does not contain the correct number of variables"
            for m in matches:
                if m.lower() == "{text}":
                    extracted_text = extracted_text.replace(m, "{text}")
                elif m.lower() == "{output}":
                    extracted_text = extracted_text.replace(m, "{output}")
            assert "{text}" in extracted_text and "{output}" in extracted_text, "Prompt does not contain the correct keywords"
            return extracted_text
        except AssertionError as e:
            print(e, f"Prompt: {extracted_text}")
            print(f"Generating new scoring prompt. Attempt {i+1} failed.")
        
    raise ValueError("Scoring Prompt could not be generated.")


async def score(prompts, testing_sample, scoring_prompt):
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
        generated_response = await run_llm_coroutine(prompt_interpolated, temperature=0.0, msg="Scoring - 30 calls mostly")
        scoring_prompt_interpolated = [scoring_prompt.format(output=generated_response[i], text=data_pair["text"]) for i, data_pair in enumerate(testing_sample)]
        prompt_scores = await run_llm_coroutine(scoring_prompt_interpolated, temperature=0.0, model="llama3-70b", max_tokens=5, msg="Scoring - 30 calls mostly")
        print(prompt_scores)
        assert len(generated_response) == len(testing_sample) == len(scoring_prompt_interpolated)
        for i in range(len(generated_response)):
            accuracy += int("yes" in prompt_scores[i].lower())
        prompt_score_pairs[prompt] = accuracy / len(testing_sample) * 100

    return prompt_score_pairs


if __name__ == "__main__":
#     ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
#     lst = []
#     for i in range(250):
#         abstract = ds["train"][i]["context"]["contexts"]
#         abstract = "\n".join(abstract)
#         data_pair = {
#             "question": f"""Question: {ds["train"][i]["question"]}\nAbstract: {abstract}""",
#             "answer": ds["train"][i]["final_decision"]
#         }
#         lst.append(data_pair)
        
#     # Convert lst to dataframe
#     true_dataset = pd.DataFrame(lst)
#     # Rename the columns
#     true_dataset.columns = ["text", "response"]
#     true_dataset = true_dataset.sample(70, random_state=42)
    
#     print(true_dataset.head())

#     # Loading optimized prompt
#     initial_prompts = ["""You are a very smart biology professor. Answer the student's question concisely and clearly \
# If you do not know the answer, say so.

# Here is a question:
# {TEXT}"""]
#     optimized_prompts = ["""####
# Think step by step and clearly understand the requirement. You are a very smart biology professor. Do answer the student's question concisely and clearly. If you do not know the answer, do say so.
# When you respond, organize your thoughts in a logical and systematic manner. Combine your knowledge of biology with your critical thinking skills to provide a well-reasoned answer.

# Here is a question:
# {TEXT}
# ####"""]

    # https://github.com/icyrockcom/country-capitals/blob/master/data/country-list.csv
    df = pd.read_csv('country-list.csv')
    df = df[["country", "capital"]]
    df.columns = ["text", "response"]
    true_dataset = df.sample(70, random_state=42)
    print(true_dataset.head())

    initial_prompts = ["Where is the capital of {TEXT}?"]
    optimized_prompts = ["Act as a geography expert, think step by step, and do provide the capital of {TEXT} in response to this query, using the same language style and tone as a knowledgeable assistant."]

    # Convert to Dataset
    true_dataset = Dataset.from_pandas(true_dataset)
    
    # Get Scoring Prompt
    scoring_prompt = ""
    if os.path.exists(f"score_qa_prompt.txt"):
        with open(f"score_qa_prompt.txt", "r") as f:
            scoring_prompt = f.read()
    else:
        scoring_prompt = asyncio.run(create_scoring_prompt(initial_prompts[0], {'text': true_dataset[0]["text"], 'output': true_dataset[0]["response"]}))
        with open(f"score_qa_prompt.txt", "w") as f:
            f.write(scoring_prompt)

    
    res = asyncio.run(score(initial_prompts, true_dataset, scoring_prompt))
    assert list(res.keys()) == initial_prompts
    initial_scores_trueDataset = list(res.values())
    
    res = asyncio.run(score(optimized_prompts, true_dataset, scoring_prompt))
    assert list(res.keys()) == optimized_prompts
    optimized_scores_trueDataset = list(res.values())
    
    df = pd.DataFrame({
        "initial_prompt": initial_prompts,
        "optimized_prompt": optimized_prompts,
        "initial_true_score": initial_scores_trueDataset,
        "optimized_true_score": optimized_scores_trueDataset
    })
    
    df.to_csv("score_qa.csv", index=False)
