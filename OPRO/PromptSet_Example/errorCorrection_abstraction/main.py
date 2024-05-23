import os, json
from errorCorrection_async import errorCorrection_opro
import asyncio

# PROMPTS THAT ARE SCORED
PROMPT_LIST = [
    'Please properly punctuate the given text (without omitting a single word) and output only the resulting punctuated text. Please do not omit a single word from the original text. {TEXT}'
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
                **asyncio.run(errorCorrection_opro(prompt, str(ID)))
            }

        # Save Prompt Scores
        with open(TESTING_SCORES_PATH, "w") as f:
            json.dump(prompt_testing_scores, f, indent=4)