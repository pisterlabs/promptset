import os, json
from summarization_async import summarization_opro
import asyncio

# PROMPTS THAT ARE SCORED
PROMPT_LIST = [
    "Please summarize the following text: PLACEHOLDER",
    "You are an onboarding chatboat that's very friendly and methodical. You read PLACEHOLDER and summarise the current project"
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
                **asyncio.run(summarization_opro(prompt, str(ID), TRAINING_SAMPLE_SIZE=10, TESTING_SAMPLE_SIZE=30, STEP_COUNT=5, PROMPTS_PER_STEP=5, MAX_PROMPT_SCORE_PAIRS=20))
            }

        # Save Prompt Scores
        with open(TESTING_SCORES_PATH, "w") as f:
            json.dump(prompt_testing_scores, f, indent=4)