import sys, os, json, re, random
import pandas as pd
from tqdm.auto import tqdm, trange
import torch
from llm_async import run_llm_coroutine
import sys

INTERPOLATE_VAR = "{TEXT}"
PWD = "./"

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

Act like a highly skilled prompt engineer. Your task is to create the best prompt possible using the list 26 principles from the list above.

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
    new_prompts = [CHOSEN_PROMPT]  # The SEED PROMPTS INCLUDES THE CHOSEN PROMPT
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

        responses = await run_llm_coroutine(prompts, temperature=1.0, model="llama3-70b", msg="Generating Seed Prompts - 20 calls")
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
    pattern2 = "PLACEHOLDER"
    matches1 = re.findall(pattern1, prompt)
    condition1 = len(matches1) == 1 
    condition2 = prompt.count(pattern2) == 1
    
    if not condition1 and not condition2:
        print(prompt)
    
    # Reformat the prompt
    if condition1:
        return prompt.replace(matches1[0], "{TEXT}")
    elif condition2:
        return prompt.replace(pattern2, "{TEXT}")
    
    raise ValueError("Invalid prompt format. Prompt must contain some str/var to be interpolated.")


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
Please generate a text and response pair for the prompt. Text is to be interpolated in the prompt. Response is the expected response to the prompt with the text interpolated.
Ensure that the text is delimited by <BEGIN_TEXT> and <END_TEXT> and the response is delimited by <BEGIN_RESPONSE> and <END_RESPONSE>. Generate text and response that is in the format shown below and highly relevant to the prompt. Take a deep breath and think step-by-step.

## Example Format:
<BEGIN_PROMPT> This is the prompt provided <END_PROMPT>
<BEGIN_TEXT> This is the text to be interpotated into the prompt. <END_TEXT>
<BEGIN_RESPONSE> The response to be generated from the text-interpolated prompt. <END_RESPONSE>

## Query:
<BEGIN_PROMPT> {CHOSEN_PROMPT} <END_PROMPT>
"""
        data_pairs = []
        unique_data = set()

        pbar = tqdm(total=request_count, desc="Generating Synthetic Data")
        attempt_count = 0
        while len(data_pairs) < request_count and attempt_count < 50:
            attempt_count += 1
            print(f"Attempt {attempt_count} made.")
            data_gen_prompt = SYNTH_DATA_GEN_PROMPT.format(CHOSEN_PROMPT=CHOSEN_PROMPT)
            response = await run_llm_coroutine([data_gen_prompt for _ in range(request_count)], temperature=1.2, model="llama3-70b", msg="Generating Synthetic Data - 100 calls")
            for res in response:
                print(res)
                try:
                    # Checking if the response is valid
                    text_match = re.search(r"<BEGIN_TEXT>([\s\S]*?)<END_TEXT>", res)
                    response_match = re.search(r"<BEGIN_RESPONSE>([\s\S]*?)<END_RESPONSE>", res)
                    assert text_match is not None and response_match is not None, "Invalid response format."
                    text = text_match.group(1).strip()
                    response = response_match.group(1).strip()
                    assert text not in unique_data, "Data already exists in the set."
                    unique_data.add(text)
                    data_pairs.append({"text": text, "response": response})
                    pbar.update(1)
                except Exception as e:
                    print(e)
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
instruction has "{INTERPOLATE_VAR}" so the user can replace it with the text to be interpolated. Think step by step. 
Generate only the text. Do not include the scores. Delimit the your suggested text by <BEGIN_ANSWER> and </END_ANSWER>.
"""
    new_prompts = []
    pbar = tqdm(total=request_count, desc="Optimizing")
    while len(new_prompts) < request_count:
        responses = await run_llm_coroutine([prompt for _ in range(request_count)], temperature=1.0, model="llama3-70b", msg="Optimizing - 20 calls")
        for res in responses:
            try:
                match = re.search(r'<BEGIN_ANSWER>(.*?)</END_ANSWER>', res, re.DOTALL)
                assert match is not None, "No match found for <BEGIN_ANSWER> and </END_ANSWER> tags"
                extracted_text = match.group(1)
                assert has_correct_keywords(extracted_text), "Extracted text does not have correct keywords"
                new_prompts.append(extracted_text)
                pbar.update(1)
            except Exception as e:
                # print(e)
                continue
    pbar.close()
    return new_prompts[:request_count]


# async def score(prompts, testing_sample):
#     """
#     Score the instruction using the sample.

#     Args:
#     instruction: str
#     sample: Dataset with "text" and "response" as keys

#     Returns:
#     accuracy: float
#     """
#     scoring_prompt_template = """You are an AI trained to evaluate the quality of responses to prompts.

# You will be given a prompt, an example response and an actual response. Your task is to assess the quality of the actual response in relation to the prompt.

# Please use the following scale for your evaluation:
# - "good" if the response perfectly answers the prompt.
# - "bad" if the response does not answer the prompt well.

# Be strict when evaluating the actual response. Only respond with "good" if there aren't any better possible responses to the prompt.
# Use the information provided in the prompt and example response to evaluate the actual response. The actual response should be judged based on its accuracy, relevance, and coherence. It need not be semantically identical to the example response, but it should address the same core ideas.
# Hint: Consider the relevance, coherence, and correctness of the response. 

# The prompt and responses pair will be delimited by "####". 

# Here are a few examples:

# ####
# Prompt: What is the capital of France?
# Example Response: The capital of France is Paris.
# Actual Response: The capital of France is Paris.
# ####
# Your output should be: good

# ####
# Prompt: Can you explain the theory of relativity? 
# Example Response: The theory of relativity, developed by Albert Einstein, is a fundamental concept in modern physics that has revolutionized our understanding of space, time, and gravity. In essence, the theory states that the laws of physics are the same everywhere in the universe and that the passage of time and the length of objects can vary depending on their speed and position in a gravitational field. Specifically, special relativity reveals that time appears to slow down and objects appear shorter to an observer when they are in motion relative to the observer, while general relativity shows that gravity is not a force, but rather the curvature of spacetime caused by massive objects, which warps the fabric of spacetime and affects the motion of other objects.
# Actual Response: The theory of relativity, proposed by Albert Einstein, states that the laws of physics are the same for all non-accelerating observers. It also introduced the concept of space-time.
# ####
# Output: good
# (This actual response is good, but it could be improved by providing more detail or examples.)

# ####
# Prompt: Who wrote the novel "1984"?
# Example Response: The novel "1984" was written by George Orwell.
# Actual Response: It was written by a British author.
# ####
# Output: bad
# (This actual response is bad, but it lacks detail. The name of the author, George Orwell, is missing.)

# ####
# Prompt: What is photosynthesis?
# Example Response: Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose molecules. This process involves the absorption of light by chlorophyll, a green pigment found in chloroplasts, and the subsequent conversion of carbon dioxide and water into glucose and oxygen. Photosynthesis is essential for life on Earth as it produces oxygen and provides a source of energy for organisms that cannot produce their own food.
# Actual Response: It's a process related to plants.
# ####
# Output: bad
# (This actual response is bad because it barely answers the prompt and lacks any meaningful detail.)

# ####
# Prompt: How many planets are there in our solar system?
# Example Response: There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.
# Actual Response: Shakespeare wrote many plays.
# ####
# Output: bad
# (This actual response is bad because it does not answer the prompt at all.)

# Now, let's try with a new pair:

# ####
# Prompt: {prompt}
# Example Response: {example_response}
# Actual Response: {actual_response}
# ####

# Respond with either "good" or "bad". Nothing else should be included in the output.
# """
#     prompt_score_pairs = {}
#     for prompt in tqdm(prompts, desc="Scoring"):
#         accuracy = 0
#         prompt_interpolated = [prompt.format(TEXT=data_pair["text"]) for data_pair in testing_sample]
#         generated_response = await run_llm_coroutine(prompt_interpolated, temperature=0.0, msg="Scoring - 30 calls mostly")
#         assert len(generated_response) == len(testing_sample)
        
#         # Scoring the responses for the interpolated prompts using an LLM as a judge
#         scoring_prompts = []
#         for i in range(len(generated_response)):
#             scoring_prompt = scoring_prompt_template.format(prompt=prompt, example_response=testing_sample[i]["response"], actual_response=generated_response[i])
#             scoring_prompts.append(scoring_prompt)
#         scoring_responses = await run_llm_coroutine(scoring_prompts, temperature=0.0, max_tokens=2, model="llama3-70b")
        
#         # Prompt the LLM to rescore for responses with improper formatting
#         scores = []
#         try_again_prompts = []
#         RESCORING_LIMIT = 10
#         for _ in range(RESCORING_LIMIT):
#             if len(scores) == len(generated_response):
#                 break
            
#             print(scores)
#             print(scoring_responses)
#             try_again_prompts = []
#             for i in range(len(scoring_responses)):
#                 output = scoring_responses[i].strip().lower()
#                 if "good" in output:
#                     scores.append(1)
#                 elif "bad" in output:
#                     scores.append(0)
#                 else:
#                     try_again_prompts.append(scoring_prompts[i])
                    
#             scoring_responses = await run_llm_coroutine(try_again_prompts, temperature=0.0, max_tokens=2, model="llama3-70b")

#         assert (len(scores) + len(try_again_prompts)) == len(generated_response)
#         accuracy = sum(scores)
#         prompt_score_pairs[prompt] = accuracy / (len(testing_sample) - len(try_again_prompts)) * 100

#     return prompt_score_pairs

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


async def opro(CHOSEN_PROMPT, training_sample, scoring_prompt, STEP_COUNT=8, PROMPTS_PER_STEP=5, MAX_PROMPT_SCORE_PAIRS=20):
    # NOTE: MAX_PROMPT_SCORE_PAIRS  Keep the best 20 prompts at any time
    SAVE_PATH_ASYNC = f"{PWD}training_results.json"
    SEED_PROMPTS_PATH = f"{PWD}seed_prompts.json"
    SEED_PROMPTS_COUNT = 16
    SEED_PROMPTS = None
    best_scores = []
    
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
        prompt_score_pairs = await score(SEED_PROMPTS, training_sample, scoring_prompt)
        # Sort by score
        prompt_score_pairs = dict(
            sorted(
                prompt_score_pairs.items(), key=lambda x: x[1], reverse=True
            )
        )
        best_scores.append(max(prompt_score_pairs.values()))
        results = {"0": prompt_score_pairs}
        with open(SAVE_PATH_ASYNC, "w") as f:
            json.dump(results, f)

    # Each step takes aboy 5 to 10 minutes with gemma:2b
    for i in range(1, STEP_COUNT + 1):
        # If the max score is reached, exit
        if max(prompt_score_pairs.values()) == 100:
            print("Max score reached. Exiting...")
            print(f"Current Best score: {max(prompt_score_pairs.values())}")
            print(f"Current Best prompt: {max(prompt_score_pairs, key=prompt_score_pairs.get)}")
            print("\n")
            break
        
        # Continue optimizing
        print(f"Step {i}")
        while True:
            try:
                # Optimizer LLM
                instructions = await opt_llm(prompt_score_pairs, request_count=PROMPTS_PER_STEP)

                # Scoring the new instructions
                new_ins_score_pairs = await score(instructions, training_sample, scoring_prompt)
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
                
                # Printing the best prompt and score for the current step
                best_prompt = max(prompt_score_pairs, key=prompt_score_pairs.get)
                best_score = prompt_score_pairs[best_prompt]
                best_scores.append(best_score)

                print(f"Step {i} completed.")
                print(f"Current Best score: {best_score}")
                print(f"Current Best prompt: {best_prompt}")
                print("\n")

                break
            except ValueError as e:
                print(e)
            except Exception as e:
                print(e)
        
        # Early stopping if the score doesn't improve for 3 consecutive steps

        if len(best_scores) > 3:
            print("Best Scores: ", best_scores[-3:])
            if best_scores[-1] == best_scores[-2] == best_scores[-3]:
                print("Early stopping...")
                break
    
    return results

# OPRO for summarization prompts
async def qarefinement_opro(prompt, cache_dir="0", TRAINING_SAMPLE_SIZE=30, TESTING_SAMPLE_SIZE=70, PROMPTS_PER_STEP=20, STEP_COUNT=15, MAX_PROMPT_SCORE_PAIRS=10):
    global PWD, CHOSEN_PROMPT
    CHOSEN_PROMPT = check_and_reformat(prompt)
    PWD = os.path.join(".", cache_dir) + "/"
    
    # If dir doesn't exist, create it
    if not os.path.exists(PWD):
        os.mkdir(cache_dir)
        
    # Open the file and set sys.stdout to the file object
    sys.stdout = open(f'{PWD}logs.txt', 'w')

    # Generate synthetic data
    synthetic_data = await generate_synthetic_data(
        CHOSEN_PROMPT, sample_size=TRAINING_SAMPLE_SIZE + TESTING_SAMPLE_SIZE
    )

    # Train-Test Split
    training_sample = synthetic_data[:TRAINING_SAMPLE_SIZE]
    testing_sample = synthetic_data[
        TRAINING_SAMPLE_SIZE : TRAINING_SAMPLE_SIZE + TESTING_SAMPLE_SIZE
    ]
    
    # Get Scoring Prompt
    scoring_prompt = ""
    if os.path.exists(f"{PWD}scoring_prompt.txt"):
        with open(f"{PWD}scoring_prompt.txt", "r") as f:
            scoring_prompt = f.read()
    else:
        scoring_prompt = await create_scoring_prompt(CHOSEN_PROMPT, {'text': testing_sample[0]["text"], 'output': testing_sample[0]["response"]})
        with open(f"{PWD}scoring_prompt.txt", "w") as f:
            f.write(scoring_prompt)
    
    # OPRO
    opro_results = await opro(CHOSEN_PROMPT, training_sample, scoring_prompt, STEP_COUNT=STEP_COUNT, PROMPTS_PER_STEP=PROMPTS_PER_STEP, MAX_PROMPT_SCORE_PAIRS=MAX_PROMPT_SCORE_PAIRS)
    best_prompt = max(opro_results[str(len(opro_results)-1)], key=opro_results[str(len(opro_results)-1)].get)

    # Comparing the initial prompt with the optimized prompt
    print("Calculating Test Scores...")
    result = {
        "initial_prompt": await score([CHOSEN_PROMPT], testing_sample, scoring_prompt),
        "optimized_prompt": await score([best_prompt], testing_sample, scoring_prompt),
    }

    # Printing Test Scores
    print("Printing Test Scores:")
    print(f"Initial Prompt Score: {result['initial_prompt']}")
    print(f"Optimized Prompt Score: {result['optimized_prompt']}")
    
    # Reset sys.stdout to its original state
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    return result
