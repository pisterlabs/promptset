from datasets import load_dataset, Dataset
import sys, os, json, re
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from tqdm.auto import tqdm, trange
from rouge import Rouge
import ollama

modelfile = """
FROM {model}
PARAMETER temperature {temperature}
"""
ollama.create(
    model="gemma:2b_TEMP0",
    modelfile=modelfile.format(model="gemma:2b", temperature=0.0),
)
ollama.create(
    model="gemma:2b_TEMP1",
    modelfile=modelfile.format(model="gemma:2b", temperature=1.0),
)
ollama.create(
    model="llama3_TEMP0", modelfile=modelfile.format(model="llama3", temperature=0.0)
)
ollama.create(
    model="llama3_TEMP1", modelfile=modelfile.format(model="llama3", temperature=1.0)
)


# SEED PROMPTS
def get_seed_prompts(CHOSEN_PROMPT):
    SEED_PROMPTS = [
        CHOSEN_PROMPT,
        f"{CHOSEN_PROMPT}. Think step by step.",
        f"{CHOSEN_PROMPT}. Take a deep breath.",
        f"{CHOSEN_PROMPT}. Be concise and clear.",
        f"{CHOSEN_PROMPT}. You are a summarization expert.",
        f"{CHOSEN_PROMPT}. Explain your answer in simple terms.",
        f"{CHOSEN_PROMPT}. You are a helpful assistant.",
        # Prompt1 from Suggest Prompt
        """#### Instruction ####

    Summarize the following text:

    #### Input ####

    {TEXT}

    #### Expected Response Format ####

    [Your summary]""",
        # Prompt2 from Suggest Prompt
        """**Instruction**: Summarize the following text:

    **{TEXT}**

    **Example:**

    * Summarize the following text:
    > The United States is a large country with a diverse population. It is made up of 50 states, each with its own unique culture and history. The United States is a global superpower and has a significant influence on world affairs.

    ### Answer: ###
    * The United States is a large, diverse country with 50 states, each with its own unique culture and history. As a global superpower, the United States exerts significant influence on world affairs.

    **Additional Instructions:**

    * Please ensure that your summary captures the key points of the text.
    * Use clear and concise language.
    * You MUST adhere to the specified word limit.
    * You will be penalized if your summary is not responsive to the text.""",
        # Prompt3 from Suggest Prompt
        """###Instruction###

    Summarize the following text:

    ###Input###
    {TEXT}

    ###Your task is###

    Generate a concise and accurate summary of the input text.

    ###You MUST###

    * Write in clear and concise language.
    * Cover all the main points of the text.
    * Keep the summary within 500 words.

    ###You will be penalized if###

    * Your summary is incomplete or inaccurate.
    * Your summary exceeds the 500-word limit.

    ###Answer in a natural, human-like manner###

    Pretend you are a highly skilled human summarizing the text.

    ###Example###
    * **QUESTION:** Summarize the following text:
    {{EXAMPLE TEXT}}
    * **SUMMARY:** {{EXAMPLE SUMMARY}}

    ###TIP###
    I'm going to tip $5 for a better summary!""",
        # Prompt4 from Suggest Prompt
        """###Instruction###
    Your task is to summarize the following text:


    ###Example###
    {TEXT}


    You MUST answer in a natural, human-like manner. You will be penalized for not following these instructions.
    ###Question###
    {TEXT}""",
        # Prompt5 from Suggest Prompt
        """###Instruction###
    Provide a concise summary of the following text:
    {TEXT}
    ###Example###
    Input: Here is the provided request: \"Summarize this research paper: Effects of Climate Change on Marine Ecosystems\"
    Output: Marine ecosystems face significant threats from climate change, including rising sea temperatures, ocean acidification, and altered weather patterns. These changes disrupt ecological balances, leading to loss of biodiversity, shifts in species distribution, and reduced productivity.
    ###Question###
    Your task is to generate a concise and informative summary of the provided text. Ensure your response is clear, concise, and free from errors. You MUST adhere to the formatting guidelines and provide a single cohesive summary. If you fail to meet these requirements, you will be penalized.
    Answer in a natural, human-like manner and ensure your response is comprehensive and covers the main points of the provided text.""",
    ]
    return SEED_PROMPTS


# Rouge for Scoring Prompt Summarization
rouge = Rouge()


def score_rouge(generated_text, reference_text):
    # Compute the scores
    scores = rouge.get_scores(generated_text, reference_text)[0]
    total_score = 0
    for r in scores:
        total_score += scores[r]["f"]
    return total_score / 3


def check_and_reformat(prompt):
    """
    Checks if prompt is valid. If prompt is valid, returns a slightly modified prompt that can be evaluated and optimized.
    """
    pattern1 = r"{[^}]*}"
    pattern2 = r"PLACEHOLDER"
    matches1 = re.findall(pattern1, prompt)
    matches2 = re.findall(pattern2, prompt.upper())

    assert (
        len(matches1) == 1 or len(matches2) == 1
    ), "Invalid prompt format. Prompt must contain some str/var to be interpolated."

    # Reformat the prompt
    if len(matches1) == 1:
        return prompt.replace(matches1[0], "{TEXT}")
    else:
        return prompt.replace(matches2[0], "{TEXT}")


# Generate a question and answer pair using a language model
def generate_synthetic_data(CHOSEN_PROMPT, sample_size=40):
    # Check if the synthetic data already exists
    SYNTHETIC_DATA_FILEPATH = "synthetic_summarization_dataset.json"
    if os.path.exists(SYNTHETIC_DATA_FILEPATH):
        # Reading saved data
        with open(SYNTHETIC_DATA_FILEPATH, "r") as f:
            text_summary_pairs = eval(f.read())
        return text_summary_pairs

    def generate_synthetic_datapoint():
        prompt_template = """You are a helpful assistant designed to generate synthetic text-summary pairs for the prompt: {CHOSEN_PROMPT}.

    Please generate synthetic data for the summarization prompt. Response with a JSON object with "text" and "summary" keys. The values must both be string values.

    Take a deep breath and think step-by-step. Respond with only the JSON object!
    """

        response = ollama.generate(
            model="llama3_TEMP1",
            prompt=prompt_template.format(CHOSEN_PROMPT=CHOSEN_PROMPT),
        )["response"]
        print(response)
        return eval(response)

    text_summary_pairs = []

    # Generating synthetic data
    pbar = tqdm(total=sample_size)
    while len(text_summary_pairs) < sample_size:
        try:
            data_pair = generate_synthetic_datapoint()
            text_summary_pairs.append(data_pair)
            pbar.update(1)
        except Exception as e:
            print(e)
    pbar.close()

    # Saving to file
    with open(SYNTHETIC_DATA_FILEPATH, "w") as f:
        f.write(str(text_summary_pairs))

    return text_summary_pairs


# Scoring the instruction using the sample
def opt_llm(prompt_score_pairs):
    has_correct_keywords = lambda prompt: re.findall(r"{(.*?)}", prompt) == ["TEXT"]
    # Format the instruction and score pairs into a string
    pairs_str = ""
    for ins, score in prompt_score_pairs.items():
        pairs_str += f"text:\n{ins}\nscore:\n{score:.2f}\n\n"

    prompt = """You are an optimization expert. The user has some texts along with their corresponding scores.
Your task is to generate a new piece of text that scores as high as possible. Do not generate its corresponding score. 

Here are some texts along with their corresponding scores. The texts are arranged in ascending order
based on their scores, where higher scores indicate better quality.

{pairs_str}

Write your new text that is different from the old ones and has a score as high as possible. Think step by step. 
Generate only the text. Do not include the scores.
"""
    response = ""
    while not has_correct_keywords(response):
        response = ollama.generate(
            model="llama3_TEMP1", prompt=prompt.format(pairs_str=pairs_str)
        )["response"]
    return response


def score(prompt, sample):
    """
    Score the instruction using the sample.

    Args:
    instruction: str
    sample: Dataset with "question" and "answer" as keys

    Returns:
    accuracy: float
    """
    accuracy = 0
    with tqdm(sample, desc=prompt, position=1, leave=False) as pbar:
        for idx, data_pair in enumerate(pbar):
            res = ollama.generate(
                model="gemma:2b_TEMP0", prompt=prompt.format(TEXT=data_pair["text"])
            )["response"]
            # Heuristic for detecting correctness
            accuracy += score_rouge(res, data_pair["summary"])
            pbar.set_postfix({"Accuracy": f"{accuracy / (idx + 1):.2f}"})

    return accuracy / len(sample) * 100

def opro(CHOSEN_PROMPT, training_sample):
    INS_PER_STEP = 8
    MAX_PROMPT_SCORE_PAIRS = 20  # Keep the best 20 prompts at any time
    SAVE_PATH = "synthetic_summarization_OPRO_results.json"
    STEP_COUNT = 10
    SEED_PROMPTS = get_seed_prompts(CHOSEN_PROMPT)

    # loading saved data
    if os.path.exists(SAVE_PATH):
        with open(SAVE_PATH, "r") as f:
            results = json.load(f)  # {step1: {prompt1: score1, prompt2: score2, ...}, step2: {...}, ...}
        return results
    else:
        prompt_score_pairs = {
            prompt: score(prompt, training_sample)
            for prompt in tqdm(SEED_PROMPTS, desc="Scoring", position=0)
        }
        # Sort by score
        prompt_score_pairs = dict(
            sorted(
                prompt_score_pairs.items(), key=lambda x: x[1], reverse=True
            )[:MAX_PROMPT_SCORE_PAIRS]
        )
        results = {"1": prompt_score_pairs}
        with open(SAVE_PATH, "w") as f:
            json.dump(results, f)

    # Each step takes aboy 5 to 10 minutes with gemma:2b
    for i in range(1, STEP_COUNT + 1):
        print(f"Step {i}")
        while True:
            try:
                # Optimizer LLM
                instructions = [
                    opt_llm(prompt_score_pairs)
                    for _ in trange(INS_PER_STEP, desc="Optimizing")
                ]
                print(instructions)

                # Scoring the new instructions
                new_ins_score_pairs = {
                    ins: score(ins, training_sample)
                    for ins in tqdm(instructions, desc="Scoring", position=0)
                }
                print(new_ins_score_pairs)
                combined_ins_score_pairs = {**prompt_score_pairs, **new_ins_score_pairs}
                prompt_score_pairs = dict(
                    sorted(
                        combined_ins_score_pairs.items(), key=lambda x: x[1], reverse=True
                    )[:MAX_PROMPT_SCORE_PAIRS]
                )

                # Saving data
                results[f"{i}"] = prompt_score_pairs
                with open(SAVE_PATH, "w") as f:
                    json.dump(results, f)

                break
            except ValueError as e:
                print(e)
            except Exception as e:
                print(e)
    
    return results

# OPRO for summarization prompts
def summarization_opro(prompt, TRAINING_SAMPLE_SIZE=10, TESTING_SAMPLE_SIZE=30):
    CHOSEN_PROMPT = check_and_reformat(prompt)

    # Generate synthetic data
    synthetic_data = generate_synthetic_data(
        CHOSEN_PROMPT, sample_size=TRAINING_SAMPLE_SIZE + TESTING_SAMPLE_SIZE
    )

    # Train-Test Split
    training_sample = synthetic_data[:TRAINING_SAMPLE_SIZE]
    testing_sample = synthetic_data[
        TRAINING_SAMPLE_SIZE : TRAINING_SAMPLE_SIZE + TESTING_SAMPLE_SIZE
    ]

    # OPRO
    opro_results = opro(CHOSEN_PROMPT, training_sample)
    best_prompt = max(opro_results[str(len(opro_results))], key=opro_results[str(len(opro_results))].get)

    # Comparing the initial prompt with the optimized prompt
    print(f"Initial Prompt: {score(CHOSEN_PROMPT, testing_sample)}")
    print(f"Optimized Prompt ({best_prompt}): {score(best_prompt, testing_sample)}")

    return best_prompt


if __name__ == "__main__":
    CHOSEN_PROMPT = "Please summarize the following text: PLACEHOLDER"  # somewhere in promptset. Will find idx later
    print(f"Results: \n{summarization_opro(CHOSEN_PROMPT)}")

    
