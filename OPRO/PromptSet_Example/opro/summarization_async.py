from datasets import load_dataset, Dataset
import sys, os, json, re, random
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
from tqdm.auto import tqdm, trange
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util
from llm_async import run_llm_coroutine

INTERPOLATE_VAR = "{TEXT}"
PWD = "./"

# Load transformer model
transformer_model = SentenceTransformer("all-mpnet-base-v2")  # Load transformer model
def similarity(text1, text2):
        embeddings = transformer_model.encode([text1, text2])
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()


# 26 prompt principles
PROMPT_PRINCIPLES = """No need to be polite with LLM so there is no need to add phrases like "please", "if you don't mind", "thank you", "I would like to", etc., and get straight to the point.
####Integrate the intended audience in the prompt, e.g., the audience is an expert in the field.
####Break down the complex tasks into a sequence of simpler prompts in an interactive conversation.
####Employ affirmative directives such as 'do,' while steering clear of negative language like 'don't'.
####When you need clarity or a deeper understanding of a topic, idea, or any piece of information, utilize the following prompts:
        - Explain [insert specific topic] in simple terms.
        - Explain to me like I'm 11 years old.
        - Explain to me as if I'm a beginner in [field].
        - Write the [essay/text/paragraph] using simple English like you're explaining something to a 5-year-old.
####Add "I'm going to tip $xxx for a better solution!"
####Implement example-driven prompting (Use few-shot prompting).
####When formatting your prompt, start with '###Instruction###', followed by either '###Example###* or '###Question###' if relevant. Subsequently, present your content. Use one or more line breaks to separate instructions, examples, questions, context, and input data.
####Incorporate the following phrases: "Your task is" and "You MUST".
####Incorporate the following phrases: "You will be penalized".
####Use the phrase "Answer a question given in a natural, human-like manner" in your prompts.
####Use leading words like writing "think step by step”.
####Add to your prompt the following phrase "Ensure that your answer is unbiased and does not rely on stereotypes".
####Allow the model to elicit precise details and requirements from you by asking you questions until he has enough information to provide the needed output (for example, "From now on, I would like you to ask me questions to...").
####To inquire about a specific topic or idea or any information and you want to test your understanding, you can use the following phrase: "Teach me the [Any theorem/topic/rule name] and include a test at the end, but don't give me the answers and then tell me if I got the answer right when I respond".
####Assign a role to the large language models.
####Use Delimiters.
####Repeat a specific word or phrase multiple times within a prompt.
####Combine Chain-of-thought (CoT) with few-Shot prompts.
####Use output primers, which involve concluding your prompt with the beginning of the desired output. Utilize output primers by ending your prompt with the start of the anticipated response.
####To write an essay/text/paragraph/article or any type of text that should be detailed: "Write a detailed [essay/text/paragraph] for me on [topic] in detail by adding all the information necessary".
####To correct/change specific text without changing its style: "Try to revise every paragraph sent by users. You should only improve the user's grammar and vocabulary and make sure it sounds natural. You should not change the writing style, such as making a formal paragraph casual".
####When you have a complex coding prompt that may be in different files: "From now and on whenever you generate code that spans more than one file, generate a [programming language ] script that can be run to automatically create the specified files or make changes to existing files to insert the generated code. [your question]".
####When you want to initiate or continue a text using specific words, phrases, or sentences, utilize the following prompt:
        - I'm providing you with the beginning [song lyrics/story/paragraph/essay...]: [Insert lyrics/words/sentence]'. Finish it based on the words provided. Keep the flow consistent.
####Clearly state the requirements that the model must follow in order to produce content, in the form of the keywords, regulations, hint, or instructions
####To write any text, such as an essay or paragraph, that is intended to be similar to a provided sample, include the following instructions:
        - Please use the same language based on the provided paragraph[/title/text/essay/answer].
"""
PRINCIPLE_SAMPLE_COUNT = 5
principles = PROMPT_PRINCIPLES.split('####')
random.seed(42)
    

# SEED PROMPTS
async def get_seed_prompts(CHOSEN_PROMPT, request_count=5):
    prompt_template = """Here are {PRINCIPLE_SAMPLE_COUNT} prompt principles:

{sampled_principles}

Act like a highly skilled prompt engineer. Your task is to create the best prompt possible using the prompting principles listed above.

Follow these tasks step-by-step:

Step 1: Read the entire list of 26 prompt principles. Analyze and explain each one of those 26 prompting principles.

Step 2: Create a prompt using those 26 prompting principles for the following prompt that's delimited by "####". Like the following prompt, make sure the new prompt contains exactly one interpolable variable "{INTERPOLATE_VAR}".

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
    has_correct_keywords = lambda prompt: re.findall(r"{(.*?)}", prompt) == [INTERPOLATE_VAR[1:-1]]
    new_prompts = [CHOSEN_PROMPT, "Please summarize the following text: {TEXT}\n Here's the suummary:"]  # The SEED PROMPTS INCLUDES THE CHOSEN PROMPT
    pbar = tqdm(total=request_count, desc="Generating Seed Prompts")
    pbar.update(1)  # Update the progress bar for the chosen prompt
    while len(new_prompts) < request_count:
        prompts = []
        for _ in range(request_count):
            # Sampling and interpolating 5-randomly sampled principles
            selected_principles_lst = random.sample(principles, PRINCIPLE_SAMPLE_COUNT)
            sampled_principles = ""
            for i, principle in enumerate(selected_principles_lst):
                sampled_principles += f"{i+1}. {principle}"
            prompt = prompt_template.format(
                PRINCIPLE_SAMPLE_COUNT=PRINCIPLE_SAMPLE_COUNT,
                sampled_principles=sampled_principles,
                CHOSEN_PROMPT=CHOSEN_PROMPT,
                INTERPOLATE_VAR=INTERPOLATE_VAR
            )
            prompts.append(prompt)

        responses = await run_llm_coroutine(prompts, temperature=1.0, model="llama3-70b")
        for res in responses:
            try:
                new_prompt = eval(res)["step2"]
                assert has_correct_keywords(new_prompt)
                new_prompt.format(TEXT="PLACEHOLDER")  # Check if the prompt is valid
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
    ), f"Invalid prompt format. Prompt must contain some str/var to be interpolated. Prompt:\n{prompt}"

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
            text_summary_pairs = json.load(f)
        return text_summary_pairs

    async def generate_synthetic_datapoint(request_count):
        SYNTH_DATA_GEN_PROMPT = f"""You are a helpful assistant designed to generate synthetic text-summary pairs for the prompt: {CHOSEN_PROMPT}.

    Please generate synthetic data for the summarization prompt. Response with a JSON object with "text" and "summary" keys. The values must both be string values.

    Generate erroneous text that is different from the example.
    Take a deep breath and think step-by-step. Respond with only the JSON object! Nothing but JSON.
    """
        data_pairs = []
        unique_data = set()

        pbar = tqdm(total=request_count, desc="Generating Synthetic Data")
        while len(data_pairs) < request_count:
            response = await run_llm_coroutine([SYNTH_DATA_GEN_PROMPT for _ in range(request_count)], temperature=1.0, model="llama3-70b")
            for res in response:
                try:
                    # Checking if the response is valid
                    data = json.loads(res)
                    assert data["text"] not in unique_data and data["summary"] not in unique_data
                    unique_data.add(data["text"])
                    unique_data.add(data["summary"])
                    data_pairs.append(data)
                    pbar.update(1)
                except Exception as e:
                    # print(e)
                    continue
        pbar.close()
        return data_pairs[:request_count]

    # Generating synthetic data
    text_summary_pairs = await generate_synthetic_datapoint(sample_size)

    # Saving to file as json
    with open(SYNTHETIC_DATA_FILEPATH_ASYNC, "w") as f:
        json.dump(text_summary_pairs, f)

    return text_summary_pairs


# Scoring the instruction using the sample
# Scoring the instruction using the sample
async def opt_llm(prompt_score_pairs, request_count=8):
    has_correct_keywords = lambda prompt: re.findall(r"{(.*?)}", prompt) == [INTERPOLATE_VAR[1:-1]]
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
instruction has "{INTERPOLATE_VAR}" so the user can replace that with the text to be summarized. Think step by step. 
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
                # print(res)
                continue
    pbar.close()
    return new_prompts[:request_count]


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

async def opro(CHOSEN_PROMPT, training_sample, STEP_COUNT=6, PROMPTS_PER_STEP=5, MAX_PROMPT_SCORE_PAIRS=20):
    # NOTE: MAX_PROMPT_SCORE_PAIRS  Keep the best 20 prompts at any time
    SAVE_PATH_ASYNC = f"{PWD}training_results.json"
    SEED_PROMPTS_PATH = f"{PWD}seed_prompts.json"
    SEED_PROMPTS_COUNT = 20
    SEED_PROMPTS = None
    
    # loading seed prompts
    if os.path.exists(SEED_PROMPTS_PATH):
        with open(SEED_PROMPTS_PATH, "r") as f:
            SEED_PROMPTS = json.load(f)
    else:
        SEED_PROMPTS = await get_seed_prompts(CHOSEN_PROMPT, request_count=SEED_PROMPTS_COUNT)
        SEED_PROMPTS = [check_and_reformat(prompt) for prompt in SEED_PROMPTS]
        with open(SEED_PROMPTS_PATH, "w") as f:
            json.dump(SEED_PROMPTS, f)

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
async def summarization_opro(prompt, cache_dir="0", TRAINING_SAMPLE_SIZE=10, TESTING_SAMPLE_SIZE=30, PROMPTS_PER_STEP=20, STEP_COUNT=10, MAX_PROMPT_SCORE_PAIRS=20):
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
    result = {
        "initial_prompt": await score([CHOSEN_PROMPT], testing_sample),
        "optimized_prompt": await score([best_prompt], testing_sample),
    }

    # Printing Test Scores
    print("Printing Test Scores:")
    print(f"Initial Prompt Score: {result['initial_prompt']}")
    print(f"Optimized Prompt Score: {result['optimized_prompt']}")

    return result
