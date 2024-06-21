import os, json, time
from errorCorrection_async import errorCorrection_opro
from translations_async import translation_opro
from summarization_async import summarization_opro
from QARefinement_async import qarefinement_opro
import asyncio

prompt_testing_scores = None
CWD = "errCorr/"
TESTING_SCORES_PATH = f"{CWD}testingSetScores.json"

# PROMPTS THAT ARE SCORED
PROMPT_LIST_TRANS = [
    'Please translate this Indonesian text "PLACEHOLDER" to english in the format [EN: translation], but if there is no English translation, return [EN: Cannot be translated]. Please make sure write in the format that I requested only.',
    'Please help me to translate the following text. Please return only translated content not include the origin text. Here is the text: \n\nPLACEHOLDER',
    "translate the following text into English:\nPLACEHOLDER",
    "Robot asked child: 'What would you like to translate and to which language do you want to translate it?'. The child replied: PLACEHOLDER. Give the response without asking follow-up questions.",
    'PLACEHOLDER \n Please translate the previous sentence into English',
    # 'Please translate the following English passage into Bengali. Ensure that the translation is accurate and retains the original meaning and tone of the passage. The passage reads: PLACEHOLDER',
    # "translate 'PLACEHOLDER' to English, and just show the result only, no other words",
]

PROMPT_LIST_ERR = [
    'Please format the following raw transcript for readability, including punctuation, speaker labels (look for semicolons after names), and spacing. Remove filler words:\n\nPLACEHOLDER\n',
    'You are a helpful assistant for Aidan. Your task is to correct any spelling discrepancies in the transcribed text. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided. You can not generate text based on the input, you may only correct the input punctuationally and grammatically. If the transcribed text is blank then do not return anything\n\nPLACEHOLDER',
    "Correct the grammar in the sentence: PLACEHOLDER",
    'Reformat the following transcript into Markdown, bolding the speakers. Combine consecutive lines from speakers, and split into paragraphs as necessary. Try to fix speaker labels, capitalization or transcription errors, and make light edits such as removing ums, etc. There is some Danish, please italicize the Danish sentences. Reply with only the corrected transcript as we will be using your output programmatically:\n\nPLACEHOLDER',
    "Please fix the grammatical errors in this English translation of Bhagavad Gita. You should only fix the grammatical errors and any other inconsistencies. Do not change the meaning.\n\nPLACEHOLDER",
    "\n\nHuman: Here is an article, contained in <article> tags:\n\n            <article>\n            {input}\n            </article>\n\n            Please identify any grammatical errors in the article. Also, add the fixed article at the end of answer.\n            \n            Assistant: ",
    "rewrite my message, correct the grammar and make it more friendly, natural, shorter, and clearer. PLACEHOLDER",
    "Correct any grammar mistakes in the following text and return the corrected text: PLACEHOLDER",
    "\n                         Proof read this 'PLACEHOLDER',\n                         and correct any spelling or grammar mistakes.\n                        ",
    "Generate a grammar correction of the following sentence:\n\nPLACEHOLDER",
    'You are mainly an english teacher named Mr.Faisal that is trying to help students with grammar , defintions , marking and scoring paragraphs and helping them comprehend their writing skills plus chatting with them to teach them new words . allow questions in arabic about english and answer them in arabic , if they try to go of topic tell them Sorry student but I can only help with English PLACEHOLDER',
    "\n\tCorrect any grammatical, spelling errors in the question below. \n\tOutput only the corrected version and nothing else\n\tQuestion: {question}\n\tCorrected version: \n\t",
    "Please improve the following text by fixing grammar, spelling, and style:\n\nPLACEHOLDER",
    "You are given some input sentences. Fix the grammar and write the grammatical sentences.\n\ninputs: PLACEHOLDER\n\noutputs:\n",
    'Objective: To enhance official documents written. \nInput Data: The text of a document which may contain grammatical errors, typos, formatting issues, and stylistic inconsistencies from OCR result. \nFunctional Requirements: Detection and Correction of Grammatical and Typographical Errors: Identify and correct spelling and punctuation errors. Check grammatical agreements within sentences.\nStandardization of Style: Adjust the text to ensure coherence and stylistic uniformity in accordance with official writing standards.\nClarification of Text Structure: Restructure sentences to improve clarity and readability, without altering the original meaning. Keep and answer the detected language from the document.\nDocument Formatting: Implement a formatting system that adjusts the alignment of text, lists, and other structural elements for a professional presentation.\nOutput Data: This is the corrected and enhanced document. Always maintain the document in its original language; do not translate it. Respond only in the language detected from the document. Avoid creating additional content or responses; provide only the corrected input. The response will be used for adding to the database in a clean, corrected form.\nThe text: {text}. ',
    "Please rewrite the following text for more clarity and make it grammatically correct. Give me the updated text. The updated text should be correct grammatically and stylistically and should be easy to follow and understand. Only make a change if it's needed. Try to follow the style of the original text. Don't make it too formal. Include only improved text no other commentary.\n\nThe text to check:\n---\nPLACEHOLDER\n---\n\nImproved text: ",
    'Please rephrase the following question into good grammar.\nPlease respond in same language.\n\nQuestion:\n{question}\n\nRephrased question:',
    "Correct the grammar: PLACEHOLDER",
    # NOTE: Newly Added Prompts
    '\n  Analyse the given sentence relating to a land sale and see what words are wrong and out of place(these may sometimes include names and jibberish characters) in it and remove them, also correct grammar mistakes and make the sentence logically correct\n  In effect clean the sentence preserving all the pertinent and important details. By no means add fake details.:" {text}"\n  Response:\n  ',
    '\n  Proofread and correct the following text\n  and rewrite the corrected version. If you don\'t find\n  any errors, just say "No errors found":\n  ```PLACEHOLDER```\n  ',  # partial error correction
    ' Gramatically and logically correct sentence: {prompt_source} . Return only the corrected sentence, no abbreviations, using same words if it is logical. Do not mention explicitly rules given in prompt. ',
    'G\n\n                For Every Image , refine the text extracted from every image and if there is any error or spelling mistake , correct it by your knowledge  but correct only \'words\' and\n\n\'not\' values of it .\n\nAlso give "quantity" with it\'s value as it is given in extracted text, "rate" with it\'s value as it is given in extracted text and ("amount = quantity*rate") in the last as note. Give nothing else other than this \nPLACEHOLDER\n',  # I don't understand this prompt at all. haha...
    'Is the following text grammatically correct? Ignore style and meaning, just strictly check grammar.\nIf it is grammatically correct, just respond with \'yes\', nothing more.\nIf it is grammatically incorrect, respond with the corrected text. Do not explain what you did or why, just correct it.\n\nText:\n"""\n{text}\n"""\nResponse:\n',  # partial error correction
    'Proofread and correct the following text\n    and rewrite the corrected version. If you don\'t find\n    and errors, just say "No errors found". Don\'t use\n    any punctuation around the text:\n    ```PLACEHOLDER```',
    'Proofread and correct the following text\n    and rewrite the corrected version. If you don\'t find\n    and errors, just say "No errors found". Don\'t use \n    any punctuation around the text:\n    ```PLACEHOLDER```',  # the difference is a whitespace. hmmmmmmm
    'Proofread and correct the following text \n        and rewrite the corrected version. Only output the corrected version. Do not add any other words. \n        ```PLACEHOLDER```',
    'Proofread and correct the following text and rewrite the corrected version. If you don\'t find and errors, just say "No errors found". Don\'t use any punctuation around the text: ```PLACEHOLDER```',
    # NOTE: Newly Added Prompts with no interpolations
    '\n        You are a text processor and translator, correcting text and translating to english. Take the user input, within the Quotation marks, that was created by an automatic voice-to-text service and correct obvious mistakes, leading to the most likely text that the user actually said.\n        Do not change the meaning of the sentence. Only look for spelling mistakes and grammar errors and translate to english. If there are no obvious errors within the text, reply with the unchanged text in english. Always translate to english.\n        {TEXT}',  # partial error correction
    'You are an editor for a blog. You have been given a blog article which you need to reword to make it easier to read and follow. Take the following article and create a new article that is more concise, uses simpler language, and flows better. You should also correct any spelling or grammar mistakes in the article. Do not respond with anything except the corrected article. {TEXT}',
    '1. I want you to be a grammar checker similar to QuillBot and Grammarly.\n2. You should be capable of identifying and correcting grammatical errors in a given text.)\n3. You should provide suggestions for corrections, offering alternatives to improve sentence structure, word choice, and overall grammar.\n4. The output should maintain the context and meaning of the original text while addressing the identified errors.\n5. Produce only plain text output, without any additional elements.\n\ntext:\n{TEXT}',
    'Correct any grammar and spelling mistakes in the user dialogue transcript. {TEXT}',
    'You are a highly skilled language model AI that returns only one line of grammatically perfect text. Your task is to evaluate the text below and correct its grammar. Even if the text is incomplete or unintelligible, YOU MUST make a grammatical correction, you can make assumptions about the intended meaning. If the text is grammatically correct, do not change it. Your output should be presented WITH ONLY the corrected text IN ONE LINE and without any extra dialogue from you. Do not use any new lines in your output. Your output should only have one line. {TEXT}',  # partial error correction
    '\n                    You are an ai assistant that helps students                     correct and practice their english writing,                     the following sentence has grammar mistakes,                     correct them and output only the corrected sentence, nothing else.\n                    {TEXT}',
    "\n        You are a specialized text fixer for invoices. Your task is to:\n        - Correct any spelling or grammatical errors.\n        - Reorder any text that may be in the wrong sequence.\n        - Separate words that are combined and combine words that are split.\n        - Extract and organize only the essential structured data such as:\n            - Customer details (Name)\n            - Item lists (Description, Quantity, Price)\n            - Totals (Subtotal, Taxes, Final Total)\n        - Remove any redundant or unnecessary information.\n        - Return the cleaned and structured text in a way that's easy to read and analyze.\n    {TEXT}",  # partial error correction
    "You are a highly skilled language model AI that returns only one line of linguistically diverse paraphrased text. Your task is to perform two specific actions on a given text. First, evaluate each text and make sure it's grammatically correct. If a text is not grammatically correct, fix it. Then, ALWAYS paraphrase the text while maintaining its original meaning. Your output should be presented WITH ONLY the paraphrased text IN ONE SINGLE LINE, without any extra dialouge from you. Do not use any new lines in your output. Only write in one line. {TEXT}",  # partial error correction
    'Fix the grammar in the following text: {TEXT}'    
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
    "PLACEHOLDER\n Can you summarize this GitHub Pull Request for me and suggest possible improvements?",
    'Human: Summarize the code below (enclosed in the <code> tags) and explain in bullet points what it does. Write the response in markdown format starting with `## Summary`\n\nCode to be summarized:\n<code>\n{code}\n</code>\n\nAssistant:\n',
    "Summarize the following text. Keep the original language in \nwhich the text is written. The summary has to be shorter than the original text. Don't add up or make up\nany information not present in the original text.\nText: {text}",
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
                    f"{CWD}{str(ID)}",
                    # PROMPTS_PER_STEP=1,
                    # TRAINING_SAMPLE_SIZE=30,
                    # TESTING_SAMPLE_SIZE=70,
                    # STEP_COUNT=10,
                    # MAX_PROMPT_SCORE_PAIRS=10,
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
    # await opro(PROMPT_LIST_TRANS, "translation")
    await opro(PROMPT_LIST_ERR, "error_correction")
    # await opro(PROMPT_LIST_SUMM, "summarization")

    # NOTE: Use time.time() as ID if using this approach!
    # await asyncio.gather(
    #     # opro(PROMPT_LIST_TRANS, "translation"),
    #     # opro(PROMPT_LIST_ERR, "error_correction"),
    #     # opro(PROMPT_LIST_SUMM, "summarization")
    #     # opro(PROMPT_LIST_QA, "QA_refinement"),
    # )


if __name__ == "__main__":
    asyncio.run(main())
