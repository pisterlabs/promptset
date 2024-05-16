from datasets import load_dataset, Dataset
import sys, os, json, re
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from tqdm.auto import tqdm, trange
from rouge import Rouge
from llm_async import run_llm

INTERPOLATE_VAR = "{TEXT}"
PWD = "./"

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
        f"""#### Instruction ####

    Summarize the following text:

    #### Input ####

    {INTERPOLATE_VAR}

    #### Expected Response Format ####

    [Your summary]""",
        # Prompt2 from Suggest Prompt
        f"""**Instruction**: Summarize the following text:

    **{INTERPOLATE_VAR}**

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
        f"""###Instruction###

    Summarize the following text:

    ###Input###
    {INTERPOLATE_VAR}

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

    ###TIP###
    I'm going to tip $5 for a better summary!""",
        # Prompt4 from Suggest Prompt
        f"""###Instruction###
    Your task is to summarize the following text:

    {INTERPOLATE_VAR}

    You MUST answer in a natural, human-like manner. You will be penalized for not following these instructions.""",
        # Prompt5 from Suggest Prompt
        f"""###Instruction###
    Provide a concise summary of the following text:
    {INTERPOLATE_VAR}
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
    if not (len(matches1) == 1 or len(matches2) == 1):
        print(prompt)
    
    assert (
        len(matches1) == 1 or len(matches2) == 1
    ), "Invalid prompt format. Prompt must contain some str/var to be interpolated."

    # Reformat the prompt
    if len(matches1) == 1:
        return prompt.replace(matches1[0], INTERPOLATE_VAR)
    else:
        return prompt.replace(matches2[0], INTERPOLATE_VAR)


# Generate a question and answer pair using a language model
def generate_synthetic_data(CHOSEN_PROMPT, sample_size=40):
    # Check if the synthetic data already exists
    SYNTHETIC_DATA_FILEPATH_ASYNC = f"{PWD}synthetic_dataset.json"
    if os.path.exists(SYNTHETIC_DATA_FILEPATH_ASYNC):
        # Reading saved data
        with open(SYNTHETIC_DATA_FILEPATH_ASYNC, "r") as f:
            text_summary_pairs = json.load(f.read())
        return text_summary_pairs

    def generate_synthetic_datapoint(request_count):
        SYNTH_DATA_GEN_PROMPT = f"""You are a helpful assistant designed to generate synthetic text-summary pairs for the prompt: {CHOSEN_PROMPT}.

    Please generate synthetic data for the summarization prompt. Response with a JSON object with "text" and "summary" keys. The values must both be string values.

    Take a deep breath and think step-by-step. Respond with only the JSON object!
    """
        data_pairs = []

        pbar = tqdm(total=request_count)
        while len(data_pairs) < request_count:
            response = run_llm([SYNTH_DATA_GEN_PROMPT for _ in range(request_count)], temperature=1.0)
            for res in response:
                try:
                    # Checking if the response is valid
                    data = json.loads(res)
                    data["text"], data["summary"]
                    data_pairs.append(data)
                    pbar.update(1)
                except Exception as e:
                    # print(e)
                    continue
        pbar.close()
        return data_pairs[:request_count]

    # Generating synthetic data
    text_summary_pairs = generate_synthetic_datapoint(sample_size)

    # Saving to file
    with open(SYNTHETIC_DATA_FILEPATH_ASYNC, "w") as f:
        f.write(str(text_summary_pairs))

    return text_summary_pairs


# Scoring the instruction using the sample
# Scoring the instruction using the sample
def opt_llm(prompt_score_pairs, request_count=8):
    has_correct_keywords = lambda prompt: re.findall(r"{(.*?)}", prompt) == ["TEXT"]
    # Format the instruction and score pairs into a string
    pairs_str = ""
    for ins, score in prompt_score_pairs.items():
        pairs_str += f"text:\n{ins}\nscore:\n{score:.2f}\n\n"

    prompt = f"""You are an optimization expert. The user has some texts along with their corresponding scores.
Your task is to generate a new piece of text that scores as high as possible. Do not generate its corresponding score. 

Here are some texts along with their corresponding scores. The texts are arranged in ascending order
based on their scores, where higher scores indicate better quality.

{pairs_str}

Write your new text that is different from the old ones and has a score as high as possible. Ensure that the generated 
instruction has "{INTERPOLATE_VAR}" so the user can replace that with the text to be summarized. Think step by step. 
Generate only the text. Do not include the scores. Response in JSON format where the keys are "prompt" with a string 
value of the new instruction that has a score as high as possible, and another key "explanation" with a string value 
explaining why the instruction will score high. Think step by step. Nothing but JSON. Ensure it's properly formatted.
"""
    new_prompts = []
    pbar = tqdm(total=request_count, desc="Optimizing")
    while len(new_prompts) < request_count:
        response = run_llm([prompt for _ in range(request_count)], temperature=1.0)
        for res in response:
            try:
                new_prompt = eval(res)["prompt"]
                assert has_correct_keywords(new_prompt)
                new_prompts.append(new_prompt)
                pbar.update(1)
            except Exception as e:
                # print(e)
                continue
    pbar.close()
    return new_prompts[:request_count]


def score(prompts, testing_sample):
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
        summaries_generated = run_llm(prompt_interpolated, temperature=0.0)
        assert len(summaries_generated) == len(testing_sample)
        for i in range(len(summaries_generated)):
            accuracy += score_rouge(summaries_generated[i], testing_sample[i]["summary"])
        prompt_score_pairs[prompt] = accuracy / len(testing_sample) * 100

    return prompt_score_pairs

def opro(CHOSEN_PROMPT, training_sample):
    INS_PER_STEP = 8
    MAX_PROMPT_SCORE_PAIRS = 20  # Keep the best 20 prompts at any time
    SAVE_PATH_ASYNC = f"{PWD}training_results.json"
    STEP_COUNT = 10
    SEED_PROMPTS = get_seed_prompts(CHOSEN_PROMPT)
    SEED_PROMPTS = [check_and_reformat(prompt) for prompt in SEED_PROMPTS]

    # loading saved data
    if os.path.exists(SAVE_PATH_ASYNC):
        with open(SAVE_PATH_ASYNC, "r") as f:
            results = json.load(f)  # {step0: {prompt1: score1, prompt2: score2, ...}, step1: {...}, ...}
        return results
    else:
        # Scoring the seed prompts
        prompt_score_pairs = score(SEED_PROMPTS, training_sample)
        # Sort by score
        prompt_score_pairs = dict(
            sorted(
                prompt_score_pairs.items(), key=lambda x: x[1], reverse=True
            )[:MAX_PROMPT_SCORE_PAIRS]
        )
        results = {"0": prompt_score_pairs}
        with open(SAVE_PATH_ASYNC, "w") as f:
            json.dump(results, f)

    # Each step takes aboy 5 to 10 minutes with gemma:2b
    for i in range(1, STEP_COUNT + 1):
        print(f"Step {i}")
        while True:
            try:
                # Optimizer LLM
                instructions = opt_llm(prompt_score_pairs, request_count=INS_PER_STEP)

                # Scoring the new instructions
                new_ins_score_pairs = score(instructions, training_sample)
                combined_ins_score_pairs = {**prompt_score_pairs, **new_ins_score_pairs}
                prompt_score_pairs = dict(
                    sorted(
                        combined_ins_score_pairs.items(), key=lambda x: x[1], reverse=True
                    )[:MAX_PROMPT_SCORE_PAIRS]
                )

                # Saving data
                results[f"{i}"] = prompt_score_pairs
                with open(SAVE_PATH_ASYNC, "w") as f:
                    json.dump(results, f)

                break
            except ValueError as e:
                print(e)
            except Exception as e:
                print(e)
    
    return results

# OPRO for summarization prompts
def summarization_opro(prompt, cache_dir="0", TRAINING_SAMPLE_SIZE=10, TESTING_SAMPLE_SIZE=30):
    global PWD, CHOSEN_PROMPT
    CHOSEN_PROMPT = check_and_reformat(prompt)
    PWD = os.path.join(".", cache_dir) + "/"
    
    # If dir doesn't exist, create it
    if not os.path.exists(PWD):
        os.mkdir(cache_dir)

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
    best_prompt = max(opro_results[str(len(opro_results)-1)], key=opro_results[str(len(opro_results)-1)].get)

    # Comparing the initial prompt with the optimized prompt
    result = {
        "initial_prompt": score([CHOSEN_PROMPT], testing_sample),
        "optimized_prompt": score([best_prompt], testing_sample),
    }
    return result


if __name__ == "__main__":
    # Training Scores are stored in directories corresponding to the prompt ID
    # Testing Scores are stored in a single file called testingSetScores.json
    TESTING_SCORES_PATH = f"testingSetScores.json"
    if os.path.exists(TESTING_SCORES_PATH):
        with open(TESTING_SCORES_PATH, "r") as f:
            testing_scores = json.load(f)
    else:
        testing_scores = {}
    

    CHOSEN_PROMPT = "Please summarize the following text: PLACEHOLDER"  # somewhere in promptset. Will find idx later
    print(f"Results: \n{summarization_opro(CHOSEN_PROMPT)}")

    
