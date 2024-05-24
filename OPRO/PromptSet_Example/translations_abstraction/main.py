import os, json
from translations_async import translation_opro
import asyncio

# PROMPTS THAT ARE SCORED
PROMPT_LIST = [
    'Please translate this Indonesian text "PLACEHOLDER" to english in the format [EN: translation], but if there is no English translation, return [EN: Cannot be translated]. Please make sure write in the format that I requested only.',
    'Please translate the following English passage into Bengali. Ensure that the translation is accurate and retains the original meaning and tone of the passage. The passage reads: PLACEHOLDER',
    "translate 'PLACEHOLDER' to English, and just show the result only, no other words",
    # 'Please help me to translate the following text. Please return only translated content not include the origin text. Here is the text: \n\nPLACEHOLDER',
    # "translate the following text into English:\nPLACEHOLDER",
    # "Robot asked child: 'What would you like to translate and to which language do you want to translate it?'. The child replied: PLACEHOLDER. Give the response without asking follow-up questions.",
    # 'PLACEHOLDER \n Please translate the previous sentence into English',
    # "Translate the following text from PLACEHOLDER to PLACEHOLDER: \n\nThe text  consists of text elements seperated by the following seperator: PLACEHOLDER\n\nLeave the seperator in place, and translate the text elements in between the seperators.\nDon't change the seperator itself. Dont'a add anything to the text elements, or remove anything from them.\n\nTranslate all of the text below until the (END OF TEXT) marker.):\n\nPLACEHOLDER\n\n(END OF TEXT)\n",

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
                **asyncio.run(translation_opro(prompt, str(ID), TRAINING_SAMPLE_SIZE=10, TESTING_SAMPLE_SIZE=30, STEP_COUNT=5, PROMPTS_PER_STEP=5, MAX_PROMPT_SCORE_PAIRS=20))
            }

        # Save Prompt Scores
        with open(TESTING_SCORES_PATH, "w") as f:
            json.dump(prompt_testing_scores, f, indent=4)