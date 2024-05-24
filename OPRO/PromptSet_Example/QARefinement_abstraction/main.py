import os, json
from QARefinement_async import qarefinement_opro
import asyncio

# PROMPTS THAT ARE SCORED
PROMPT_LIST = [
    'what is a fruit of color: {color}. Return the name of the fruit and nothing else:',
    'Given that someone is feeling PLACEHOLDER, what would be a good course of action or suggestion for them?',
    'Explanation of what the code does:PLACEHOLDER',
    # 'what is the city {person} is from?',
    # 'Tell me what language this is: ```PLACEHOLDER```',
    # 'what is a country with a flag that has the color: {color}. Return the name of the country and nothing else:',
    # "'PLACEHOLDER'\n Return a valid python dictionary where each key is a stock/industry name (ticker) and the contents is a detailed explanation of what that the person said",
    # 'What is the capital of {place}?',
]

if __name__ == '__main__':
    # TRAINING SCORES: stored in directories corresponding to the prompt ID (shown in testingSetScores.json)
    # TESTING SCORES: stored in a single file called testingSetScores.json
    TESTING_SCORES_PATH = f"testingSetScores.json"
    if os.path.exists(TESTING_SCORES_PATH):
        with open(TESTING_SCORES_PATH, "r") as f:
            prompt_testing_scores = json.load(f)
    else:
        prompt_testing_scores = {}
    
    for prompt in PROMPT_LIST:
        if prompt not in prompt_testing_scores:
            ID = len(prompt_testing_scores)
            prompt_testing_scores[prompt] = {
                "ID": ID,
                **asyncio.run(qarefinement_opro(prompt, str(ID), TRAINING_SAMPLE_SIZE=10, TESTING_SAMPLE_SIZE=30, STEP_COUNT=10, PROMPTS_PER_STEP=15, MAX_PROMPT_SCORE_PAIRS=20))
            }

        # Save Prompt Scores
        with open(TESTING_SCORES_PATH, "w") as f:
            json.dump(prompt_testing_scores, f, indent=4)