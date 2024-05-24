from sentence_transformers import SentenceTransformer, util
import sys, os, json, re
import pandas as pd
from tqdm.auto import tqdm, trange
import torch
from llm_async import run_llm_coroutine

INTERPOLATE_VAR = "{TEXT}"
PWD = "./"

# Load transformer model
transformer_model = SentenceTransformer("all-mpnet-base-v2")  # Load transformer model
def similarity(text1, text2):
        embeddings = transformer_model.encode([text1, text2])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

PROMPT_PRINCIPLES = """
Here are 26 prompt principles:
1. No need to be polite with LLM so there is no need to add phrases like "please", "if you don't mind", "thank you", "I would like to", etc., and get straight to the point.
2. Integrate the intended audience in the prompt, e.g., the audience is an expert in the field.
3. Break down the complex tasks into a sequence of simpler prompts in an interactive conversation.
4. Employ affirmative directives such as 'do,' while steering clear of negative language like 'don't'.
5. When you need clarity or a deeper understanding of a topic, idea, or any piece of information, utilize the following prompts:
    - Explain [insert specific topic] in simple terms.
    - Explain to me like I'm 11 years old.
    - Explain to me as if I'm a beginner in [field].
    - Write the [essay/text/paragraph] using simple English like you're explaining something to a 5-year-old.
6. Add "I'm going to tip $xxx for a better solution!"
7. Implement example-driven prompting (Use few-shot prompting).
8. When formatting your prompt, start with '###Instruction###', followed by either '###Example###* or '###Question###' if relevant. Subsequently, present your content. Use one or more line breaks to separate instructions, examples, questions, context, and input data.
9. Incorporate the following phrases: "Your task is" and "You MUST".
10. Incorporate the following phrases: "You will be penalized".
11. Use the phrase "Answer a question given in a natural, human-like manner" in your prompts.
12. Use leading words like writing "think step by step”.
13. Add to your prompt the following phrase "Ensure that your answer is unbiased and does not rely on stereotypes".
14. Allow the model to elicit precise details and requirements from you by asking you questions until he has enough information to provide the needed output (for example, "From now on, I would like you to ask me questions to...").
15. To inquire about a specific topic or idea or any information and you want to test your understanding, you can use the following phrase: "Teach me the [Any theorem/topic/rule name] and include a test at the end, but don't give me the answers and then tell me if I got the answer right when I respond".
16. Assign a role to the large language models.
17. Use Delimiters.
18. Repeat a specific word or phrase multiple times within a prompt.
19. Combine Chain-of-thought (CoT) with few-Shot prompts.
20. Use output primers, which involve concluding your prompt with the beginning of the desired output. Utilize output primers by ending your prompt with the start of the anticipated response.
21. To write an essay/text/paragraph/article or any type of text that should be detailed: "Write a detailed [essay/text/paragraph] for me on [topic] in detail by adding all the information necessary".
22. To correct/change specific text without changing its style: "Try to revise every paragraph sent by users. You should only improve the user's grammar and vocabulary and make sure it sounds natural. You should not change the writing style, such as making a formal paragraph casual".
23. When you have a complex coding prompt that may be in different files: "From now and on whenever you generate code that spans more than one file, generate a [programming language ] script that can be run to automatically create the specified files or make changes to existing files to insert the generated code. [your question]".
24. When you want to initiate or continue a text using specific words, phrases, or sentences, utilize the following prompt:
    - I'm providing you with the beginning [song lyrics/story/paragraph/essay...]: [Insert lyrics/words/sentence]'. Finish it based on the words provided. Keep the flow consistent.
25. Clearly state the requirements that the model must follow in order to produce content, in the form of the keywords, regulations, hint, or instructions
26. To write any text, such as an essay or paragraph, that is intended to be similar to a provided sample, include the following instructions:
    - Please use the same language based on the provided paragraph[/title/text/essay/answer].
"""


# SEED PROMPTS
async def get_seed_prompts(CHOSEN_PROMPT, request_count=5):
    prompt = f"""{PROMPT_PRINCIPLES}

Act like a highly skilled prompt engineer. Your task is to create the best prompt possible using the list 26 principles from the list above.

Follow these tasks step-by-step:

Step 1: Read the entire list of 26 prompt principles. Analyze and explain each one of those 26 prompting principles.

Step 2: Create a prompt using those 26 prompting principles for the following prompt that's delimited by "####". Like the following prompt, make sure the new prompt contains exactly one interpolable variable, "{INTERPOLATE_VAR}".

####
{CHOSEN_PROMPT}
####

Respond with a JSON object containing two keys "step1" and "step2", respectively mapping to the analysis and explanation to the 26 prompting principles and the prompt you created.

Example JSON object:
{{
    "step1": \"\"\"Here is the analysis and explanation for each of the 26 prompting principles...\"\"\",
    "step2": \"\"\"Think step by step...\"\"\"
}}

Take a deep breath and work on this problem step-by-step. Return only the JSON object with the keys "step1" and "step2", and nothing else. Nothing but JSON."""
    def has_correct_keywords(s):
        extract_keys = lambda x: re.findall(r'{(.*?)}', x)
        return extract_keys(s) == [INTERPOLATE_VAR[1:-1]]
    new_prompts = [CHOSEN_PROMPT]  # The SEED PROMPTS INCLUDES THE CHOSEN PROMPT
    pbar = tqdm(total=request_count, desc="Generating Seed Prompts")
    pbar.update(1)  # Update the progress bar for the chosen prompt
    while len(new_prompts) < request_count:
        responses = await run_llm_coroutine([prompt for _ in range(request_count)], temperature=1.0, model="llama3-70b")
        for res in responses:
            try:
                new_prompt = eval(res)["step2"]
                assert has_correct_keywords(new_prompt)
                new_prompts.append(new_prompt)
                pbar.update(1)
            except Exception as e:
                # print(e)
                # print(res)
                continue
    pbar.close()
    return new_prompts[:request_count]


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
async def generate_synthetic_data(CHOSEN_PROMPT, sample_size=40):
    # Check if the synthetic data already exists
    SYNTHETIC_DATA_FILEPATH_ASYNC = f"{PWD}synthetic_dataset.json"
    if os.path.exists(SYNTHETIC_DATA_FILEPATH_ASYNC):
        # Reading saved data
        with open(SYNTHETIC_DATA_FILEPATH_ASYNC, "r") as f:
            text = json.load(f)
        return text

    async def generate_synthetic_datapoint(request_count):
        SYNTH_DATA_GEN_PROMPT = """You are a helpful assistant designed to generate synthetic data for the prompt: "{CHOSEN_PROMPT}".
Please generate a text and response pair for the prompt as a JSON object.
Text is to be interpolated in the prompt. Response is the expected response to the prompt with the text interpolated.


{{
    "text": \"\"\"This is the text to be interpotated into the prompt.\"\"\",
    "response": \"\"\"The response to be generated from the text-interpolated prompt.\"\"\"
}}

Generate text and response that is in the format shown above and highly relevant to the prompt. Make sure the values to "text" and "response" are string values.
Take a deep breath and think step-by-step. Respond with only the JSON object! Nothing but JSON.

DO NOT include any from the following:
{data_pairs}

RESPOND ONLY A SINGLE text and response pair. Nothing but JSON.
"""
        data_pairs = []
        unique_data = set()

        pbar = tqdm(total=request_count, desc="Generating Synthetic Data")
        while len(data_pairs) < request_count:
            response = await run_llm_coroutine([SYNTH_DATA_GEN_PROMPT.format(CHOSEN_PROMPT=CHOSEN_PROMPT, data_pairs=data_pairs) for _ in range(request_count)], temperature=1.0, model="llama3-70b")
            for res in response:
                try:
                    # Checking if the response is valid
                    data = json.loads(res)
                    assert data["text"] not in unique_data and data["response"] not in unique_data
                    unique_data.add(data["text"])
                    unique_data.add(data["response"])
                    data_pairs.append(data)
                    pbar.update(1)
                except Exception as e:
                    # print(e)
                    # print(res)
                    continue
        pbar.close()
        return data_pairs[:request_count]

    # Generating synthetic data
    text = await generate_synthetic_datapoint(sample_size)

    # Saving to file as json
    with open(SYNTHETIC_DATA_FILEPATH_ASYNC, "w") as f:
        json.dump(text, f)

    return text


# Scoring the instruction using the sample
# Scoring the instruction using the sample
async def opt_llm(prompt_score_pairs, request_count=8):
    def has_correct_keywords(s):
        extract_keys = lambda x: re.findall(r'{(.*?)}', x)
        return extract_keys(s) == [INTERPOLATE_VAR[1:-1]]
    # Format the instruction and score pairs into a string
    pairs_str = ""
    for ins, score in prompt_score_pairs.items():
        pairs_str += f"text:\n{ins}\nscore:\n{score:.2f}\n\n"

    prompt = f"""You're a highly skilled prompt engineer and a prompt optimization expert. 
The user has some prompts along with their corresponding scores. 
Your task is to generate a new prompt that scores as high as possible. Do not generate its corresponding score. 

Here are some prompts along with their corresponding scores. The texts are arranged in ascending order
based on their scores, where higher scores indicate better quality.

{pairs_str}

Write your new text that is different from the old ones and has a score as high as possible. Ensure that the generated 
instruction has "{INTERPOLATE_VAR}" so the user can replace it with the text to be interpolated. Think step by step. 
Generate only the text. Do not include the scores. Response in JSON format where the keys are "prompt" with a string 
value of the new instruction that has a score as high as possible, and another key "explanation" with a string value 
explaining why the instruction will score high. Think step by step. Nothing but JSON. Ensure it's properly formatted.
"""
    new_prompts = []
    pbar = tqdm(total=request_count, desc="Optimizing")
    while len(new_prompts) < request_count:
        response = await run_llm_coroutine([prompt for _ in range(request_count)], temperature=1.0, model="llama3-70b")
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


async def score(prompts, testing_sample):
    """
    Score the instruction using the sample.

    Args:
    instruction: str
    sample: Dataset with "text" and "response" as keys

    Returns:
    accuracy: float
    """
    prompt_score_pairs = {}
    for prompt in tqdm(prompts, desc="Scoring"):
        accuracy = 0
        prompt_interpolated = [prompt.format(TEXT=data_pair["text"]) for data_pair in testing_sample]
        generated_response = await run_llm_coroutine(prompt_interpolated, temperature=0.0)
        assert len(generated_response) == len(testing_sample)
        for i in range(len(generated_response)):
            accuracy += similarity(testing_sample[i]["response"], generated_response[i])
        prompt_score_pairs[prompt] = accuracy / len(testing_sample) * 100

    return prompt_score_pairs

async def opro(CHOSEN_PROMPT, training_sample, STEP_COUNT=10, PROMPTS_PER_STEP=5, MAX_PROMPT_SCORE_PAIRS=20):
    # NOTE: MAX_PROMPT_SCORE_PAIRS  Keep the best 20 prompts at any time
    SAVE_PATH_ASYNC = f"{PWD}training_results.json"
    SEED_PROMPTS = await get_seed_prompts(CHOSEN_PROMPT, request_count=PROMPTS_PER_STEP)
    SEED_PROMPTS = [check_and_reformat(prompt) for prompt in SEED_PROMPTS]

    # loading saved data
    if os.path.exists(SAVE_PATH_ASYNC):
        with open(SAVE_PATH_ASYNC, "r") as f:
            results = json.load(f)  # {step0: {prompt1: score1, prompt2: score2, ...}, step1: {...}, ...}
        return results
    else:
        # Scoring the seed prompts
        prompt_score_pairs = await score(SEED_PROMPTS, training_sample)
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
                instructions = await opt_llm(prompt_score_pairs, request_count=PROMPTS_PER_STEP)

                # Scoring the new instructions
                new_ins_score_pairs = await score(instructions, training_sample)
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
                
                # Printing the best prompt
                print(f"Step {i} completed.")
                print(f"Current Best score: {max(prompt_score_pairs.values())}")
                print(f"Current Best prompt: {max(prompt_score_pairs, key=prompt_score_pairs.get)}")
                print("\n")

                break
            except ValueError as e:
                print(e)
            except Exception as e:
                print(e)
    
    return results

# OPRO for summarization prompts
async def qarefinement_opro(prompt, cache_dir="0", TRAINING_SAMPLE_SIZE=10, TESTING_SAMPLE_SIZE=30, PROMPTS_PER_STEP=8, STEP_COUNT=5, MAX_PROMPT_SCORE_PAIRS=20):
    global PWD, CHOSEN_PROMPT
    CHOSEN_PROMPT = check_and_reformat(prompt)
    PWD = os.path.join(".", cache_dir) + "/"
    
    # If dir doesn't exist, create it
    if not os.path.exists(PWD):
        os.mkdir(cache_dir)

    # Generate synthetic data
    synthetic_data = await generate_synthetic_data(
        CHOSEN_PROMPT, sample_size=TRAINING_SAMPLE_SIZE + TESTING_SAMPLE_SIZE
    )

    # Train-Test Split
    training_sample = synthetic_data[:TRAINING_SAMPLE_SIZE]
    testing_sample = synthetic_data[
        TRAINING_SAMPLE_SIZE : TRAINING_SAMPLE_SIZE + TESTING_SAMPLE_SIZE
    ]

    # OPRO
    opro_results = await opro(CHOSEN_PROMPT, training_sample, STEP_COUNT=STEP_COUNT, PROMPTS_PER_STEP=PROMPTS_PER_STEP, MAX_PROMPT_SCORE_PAIRS=MAX_PROMPT_SCORE_PAIRS)
    best_prompt = max(opro_results[str(len(opro_results)-1)], key=opro_results[str(len(opro_results)-1)].get)

    # Comparing the initial prompt with the optimized prompt
    print("Calculating Test Scores...")
    result = {
        "initial_prompt": await score([CHOSEN_PROMPT], testing_sample),
        "optimized_prompt": await score([best_prompt], testing_sample),
    }

    # Printing Test Scores
    print("Printing Test Scores:")
    print(f"Initial Prompt Score: {result['initial_prompt']}")
    print(f"Optimized Prompt Score: {result['optimized_prompt']}")
    return result
