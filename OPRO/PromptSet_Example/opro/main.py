import os, json, time
from errorCorrection_async import errorCorrection_opro
from translations_async import translation_opro
from summarization_async import summarization_opro
from QARefinement_async import qarefinement_opro
import asyncio

prompt_testing_scores = None
TESTING_SCORES_PATH = f"testingSetScores.json"

# PROMPTS THAT ARE SCORED
PROMPT_LIST_TRANS = [
    'Please translate this Indonesian text "PLACEHOLDER" to english in the format [EN: translation], but if there is no English translation, return [EN: Cannot be translated]. Please make sure write in the format that I requested only.',
    # "translate 'PLACEHOLDER' to English, and just show the result only, no other words",
    # 'Please translate the following English passage into Bengali. Ensure that the translation is accurate and retains the original meaning and tone of the passage. The passage reads: PLACEHOLDER',
    # 'Please help me to translate the following text. Please return only translated content not include the origin text. Here is the text: \n\nPLACEHOLDER',
    # "translate the following text into English:\nPLACEHOLDER",
    # "Robot asked child: 'What would you like to translate and to which language do you want to translate it?'. The child replied: PLACEHOLDER. Give the response without asking follow-up questions.",
    # 'PLACEHOLDER \n Please translate the previous sentence into English',
    # "Translate the following text from PLACEHOLDER to PLACEHOLDER: \n\nThe text  consists of text elements seperated by the following seperator: PLACEHOLDER\n\nLeave the seperator in place, and translate the text elements in between the seperators.\nDon't change the seperator itself. Dont'a add anything to the text elements, or remove anything from them.\n\nTranslate all of the text below until the (END OF TEXT) marker.):\n\nPLACEHOLDER\n\n(END OF TEXT)\n",
]

PROMPT_LIST_ERR = [
    '\n\tCorrect any grammatical, spelling errors in the question below. \n\tOutput only the corrected version and nothing else\n\tQuestion: {question}\n\tCorrected version: \n\t',
    "Please properly punctuate the given text (without omitting a single word) and output only the resulting punctuated text. Please do not omit a single word from the original text. {TEXT}",
    # 'Objective: To enhance official documents written. \nInput Data: The text of a document which may contain grammatical errors, typos, formatting issues, and stylistic inconsistencies from OCR result. \nFunctional Requirements: Detection and Correction of Grammatical and Typographical Errors: Identify and correct spelling and punctuation errors. Check grammatical agreements within sentences.\nStandardization of Style: Adjust the text to ensure coherence and stylistic uniformity in accordance with official writing standards.\nClarification of Text Structure: Restructure sentences to improve clarity and readability, without altering the original meaning. Keep and answer the detected language from the document.\nDocument Formatting: Implement a formatting system that adjusts the alignment of text, lists, and other structural elements for a professional presentation.\nOutput Data: This is the corrected and enhanced document. Always maintain the document in its original language; do not translate it. Respond only in the language detected from the document. Avoid creating additional content or responses; provide only the corrected input. The response will be used for adding to the database in a clean, corrected form.\nThe text: {text}. ',
    # 'Please fix the grammatical errors in this English translation of Bhagavad Gita. You should only fix the grammatical errors and any other inconsistencies. Do not change the meaning.\n\nPLACEHOLDER',
    # "Please rewrite the following text for more clarity and make it grammatically correct. Give me the updated text. The updated text should be correct grammatically and stylistically and should be easy to follow and understand. Only make a change if it's needed. Try to follow the style of the original text. Don't make it too formal. Include only improved text no other commentary.\n\nThe text to check:\n---\nPLACEHOLDER\n---\n\nImproved text: ",
]

PROMPT_LIST_QA = [
    # 'Given that someone is feeling PLACEHOLDER, what would be a good course of action or suggestion for them?',
    # "what is a fruit of color: {color}. Return the name of the fruit and nothing else:",
    # 'Explanation of what the code does:PLACEHOLDER',
    # 'what is the city {person} is from?',
    # 'Tell me what language this is: ```PLACEHOLDER```',
    # 'what is a country with a flag that has the color: {color}. Return the name of the country and nothing else:',
    # "'PLACEHOLDER'\n Return a valid python dictionary where each key is a stock/industry name (ticker) and the contents is a detailed explanation of what that the person said",
    # 'What is the capital of {place}?',
]

PROMPT_LIST_SUMM = [
    "You are an onboarding chatboat that's very friendly and methodical. You read PLACEHOLDER and summarise the current project",
    # "Please summarize the following text: PLACEHOLDER",
]


async def opro(prompt_list, category):
    global prompt_testing_scores, TESTING_SCORES_PATH
    opro_func = None
    match category:
        case "translation":
            opro_func = translation_opro
        case "error_correction":
            opro_func = errorCorrection_opro
        case "QA_refinement":
            opro_func = qarefinement_opro
        case "summarization":
            opro_func = summarization_opro
        case _:
            raise ValueError(f"Invalid category: {category}")

    # Optimizing Prompts
    for prompt in prompt_list:
        if prompt not in prompt_testing_scores:
            ID = len(prompt_testing_scores)
            # ID = time.time()
            prompt_testing_scores[prompt] = {
                "ID": ID,
                **await opro_func(
                    prompt,
                    str(ID),
                    # TRAINING_SAMPLE_SIZE=30,
                    # TESTING_SAMPLE_SIZE=70,
                    # STEP_COUNT=10,
                    # PROMPTS_PER_STEP=20,
                    # MAX_PROMPT_SCORE_PAIRS=20,
                ),
                "category": category,
            }

        # Save Prompt Scores
        with open(TESTING_SCORES_PATH, "w") as f:
            json.dump(prompt_testing_scores, f, indent=4)


async def main():
    global prompt_testing_scores, TESTING_SCORES_PATH
    # TRAINING SCORES: stored in directories corresponding to the prompt ID (shown in testingSetScores.json)
    # TESTING SCORES: stored in a single file called testingSetScores.json
    if os.path.exists(TESTING_SCORES_PATH):
        with open(TESTING_SCORES_PATH, "r") as f:
            prompt_testing_scores = json.load(f)
    else:
        prompt_testing_scores = {}

    # Optimizing Prompts
    # await opro(PROMPT_LIST_QA, "QA_refinement")
    await opro(PROMPT_LIST_TRANS, "translation")
    await opro(PROMPT_LIST_ERR, "error_correction")
    await opro(PROMPT_LIST_SUMM, "summarization")

    # NOTE: Use time.time() as ID if using this approach!
    # await asyncio.gather(
    #     # opro(PROMPT_LIST_TRANS, "translation"),
    #     # opro(PROMPT_LIST_ERR, "error_correction"),
    #     # opro(PROMPT_LIST_SUMM, "summarization")
    #     # opro(PROMPT_LIST_QA, "QA_refinement"),
    # )


if __name__ == "__main__":
    asyncio.run(main())
