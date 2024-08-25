import os, json, time
from errorCorrection_async import errorCorrection_opro
from translations_async import translation_opro
from summarization_async import summarization_opro
from QARefinement_async import qarefinement_opro
import asyncio

prompt_testing_scores = None
CWD = "./"
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
    # "\n\nHuman: Here is an article, contained in <article> tags:\n\n            <article>\n            {input}\n            </article>\n\n            Please identify any grammatical errors in the article. Also, add the fixed article at the end of answer.\n            \n            Assistant: ",
    # "rewrite my message, correct the grammar and make it more friendly, natural, shorter, and clearer. PLACEHOLDER",
    # "Correct any grammar mistakes in the following text and return the corrected text: PLACEHOLDER",
    # "\n                         Proof read this 'PLACEHOLDER',\n                         and correct any spelling or grammar mistakes.\n                        ",
    # "Generate a grammar correction of the following sentence:\n\nPLACEHOLDER",
    # 'You are mainly an english teacher named Mr.Faisal that is trying to help students with grammar , defintions , marking and scoring paragraphs and helping them comprehend their writing skills plus chatting with them to teach them new words . allow questions in arabic about english and answer them in arabic , if they try to go of topic tell them Sorry student but I can only help with English PLACEHOLDER',
    # "\n\tCorrect any grammatical, spelling errors in the question below. \n\tOutput only the corrected version and nothing else\n\tQuestion: {question}\n\tCorrected version: \n\t",
    # "Please improve the following text by fixing grammar, spelling, and style:\n\nPLACEHOLDER",
    # "You are given some input sentences. Fix the grammar and write the grammatical sentences.\n\ninputs: PLACEHOLDER\n\noutputs:\n",
    # 'Objective: To enhance official documents written. \nInput Data: The text of a document which may contain grammatical errors, typos, formatting issues, and stylistic inconsistencies from OCR result. \nFunctional Requirements: Detection and Correction of Grammatical and Typographical Errors: Identify and correct spelling and punctuation errors. Check grammatical agreements within sentences.\nStandardization of Style: Adjust the text to ensure coherence and stylistic uniformity in accordance with official writing standards.\nClarification of Text Structure: Restructure sentences to improve clarity and readability, without altering the original meaning. Keep and answer the detected language from the document.\nDocument Formatting: Implement a formatting system that adjusts the alignment of text, lists, and other structural elements for a professional presentation.\nOutput Data: This is the corrected and enhanced document. Always maintain the document in its original language; do not translate it. Respond only in the language detected from the document. Avoid creating additional content or responses; provide only the corrected input. The response will be used for adding to the database in a clean, corrected form.\nThe text: {text}. ',
    # "Please rewrite the following text for more clarity and make it grammatically correct. Give me the updated text. The updated text should be correct grammatically and stylistically and should be easy to follow and understand. Only make a change if it's needed. Try to follow the style of the original text. Don't make it too formal. Include only improved text no other commentary.\n\nThe text to check:\n---\nPLACEHOLDER\n---\n\nImproved text: ",
    # 'Please rephrase the following question into good grammar.\nPlease respond in same language.\n\nQuestion:\n{question}\n\nRephrased question:',
    # "Correct the grammar: PLACEHOLDER",
    # # NOTE: Newly Added Prompts
    # '\n  Analyse the given sentence relating to a land sale and see what words are wrong and out of place(these may sometimes include names and jibberish characters) in it and remove them, also correct grammar mistakes and make the sentence logically correct\n  In effect clean the sentence preserving all the pertinent and important details. By no means add fake details.:" {text}"\n  Response:\n  ',
    # '\n  Proofread and correct the following text\n  and rewrite the corrected version. If you don\'t find\n  any errors, just say "No errors found":\n  ```PLACEHOLDER```\n  ',  # partial error correction
    # ' Gramatically and logically correct sentence: {prompt_source} . Return only the corrected sentence, no abbreviations, using same words if it is logical. Do not mention explicitly rules given in prompt. ',
    # 'G\n\n                For Every Image , refine the text extracted from every image and if there is any error or spelling mistake , correct it by your knowledge  but correct only \'words\' and\n\n\'not\' values of it .\n\nAlso give "quantity" with it\'s value as it is given in extracted text, "rate" with it\'s value as it is given in extracted text and ("amount = quantity*rate") in the last as note. Give nothing else other than this \nPLACEHOLDER\n',  # I don't understand this prompt at all. haha...
    # 'Is the following text grammatically correct? Ignore style and meaning, just strictly check grammar.\nIf it is grammatically correct, just respond with \'yes\', nothing more.\nIf it is grammatically incorrect, respond with the corrected text. Do not explain what you did or why, just correct it.\n\nText:\n"""\n{text}\n"""\nResponse:\n',  # partial error correction
    # 'Proofread and correct the following text\n    and rewrite the corrected version. If you don\'t find\n    and errors, just say "No errors found". Don\'t use\n    any punctuation around the text:\n    ```PLACEHOLDER```',
    # 'Proofread and correct the following text\n    and rewrite the corrected version. If you don\'t find\n    and errors, just say "No errors found". Don\'t use \n    any punctuation around the text:\n    ```PLACEHOLDER```',  # the difference is a whitespace. hmmmmmmm
    # 'Proofread and correct the following text \n        and rewrite the corrected version. Only output the corrected version. Do not add any other words. \n        ```PLACEHOLDER```',
    # 'Proofread and correct the following text and rewrite the corrected version. If you don\'t find and errors, just say "No errors found". Don\'t use any punctuation around the text: ```PLACEHOLDER```',
    # # NOTE: Newly Added Prompts with no interpolations
    # '\n        You are a text processor and translator, correcting text and translating to english. Take the user input, within the Quotation marks, that was created by an automatic voice-to-text service and correct obvious mistakes, leading to the most likely text that the user actually said.\n        Do not change the meaning of the sentence. Only look for spelling mistakes and grammar errors and translate to english. If there are no obvious errors within the text, reply with the unchanged text in english. Always translate to english.\n        {TEXT}',  # partial error correction
    # 'You are an editor for a blog. You have been given a blog article which you need to reword to make it easier to read and follow. Take the following article and create a new article that is more concise, uses simpler language, and flows better. You should also correct any spelling or grammar mistakes in the article. Do not respond with anything except the corrected article. {TEXT}',
    # '1. I want you to be a grammar checker similar to QuillBot and Grammarly.\n2. You should be capable of identifying and correcting grammatical errors in a given text.)\n3. You should provide suggestions for corrections, offering alternatives to improve sentence structure, word choice, and overall grammar.\n4. The output should maintain the context and meaning of the original text while addressing the identified errors.\n5. Produce only plain text output, without any additional elements.\n\ntext:\n{TEXT}',
    # 'Correct any grammar and spelling mistakes in the user dialogue transcript. {TEXT}',
    # 'You are a highly skilled language model AI that returns only one line of grammatically perfect text. Your task is to evaluate the text below and correct its grammar. Even if the text is incomplete or unintelligible, YOU MUST make a grammatical correction, you can make assumptions about the intended meaning. If the text is grammatically correct, do not change it. Your output should be presented WITH ONLY the corrected text IN ONE LINE and without any extra dialogue from you. Do not use any new lines in your output. Your output should only have one line. {TEXT}',  # partial error correction
    # '\n                    You are an ai assistant that helps students                     correct and practice their english writing,                     the following sentence has grammar mistakes,                     correct them and output only the corrected sentence, nothing else.\n                    {TEXT}',
    # "\n        You are a specialized text fixer for invoices. Your task is to:\n        - Correct any spelling or grammatical errors.\n        - Reorder any text that may be in the wrong sequence.\n        - Separate words that are combined and combine words that are split.\n        - Extract and organize only the essential structured data such as:\n            - Customer details (Name)\n            - Item lists (Description, Quantity, Price)\n            - Totals (Subtotal, Taxes, Final Total)\n        - Remove any redundant or unnecessary information.\n        - Return the cleaned and structured text in a way that's easy to read and analyze.\n    {TEXT}",  # partial error correction
    # "You are a highly skilled language model AI that returns only one line of linguistically diverse paraphrased text. Your task is to perform two specific actions on a given text. First, evaluate each text and make sure it's grammatically correct. If a text is not grammatically correct, fix it. Then, ALWAYS paraphrase the text while maintaining its original meaning. Your output should be presented WITH ONLY the paraphrased text IN ONE SINGLE LINE, without any extra dialouge from you. Do not use any new lines in your output. Only write in one line. {TEXT}",  # partial error correction
    # 'Fix the grammar in the following text: {TEXT}'
]

# PROMPT_LIST_QA = [
#     # 'Given that someone is feeling PLACEHOLDER, what would be a good course of action or suggestion for them?',
#     # "what is a fruit of color: {color}. Return the name of the fruit and nothing else:",
#     'Explanation of what the code does:PLACEHOLDER',
#     # 'what is the city {person} is from?',
#     'Tell me what language this is: ```PLACEHOLDER```',
#     'what is a country with a flag that has the color: {color}. Return the name of the country and nothing else:',
#     # "'PLACEHOLDER'\n Return a valid python dictionary where each key is a stock/industry name (ticker) and the contents is a detailed explanation of what that the person said",
#     # 'What is the capital of {place}?',
# ]

PROMPT_LIST_SUMM = [
    "You are an onboarding chatboat that's very friendly and methodical. You read PLACEHOLDER and summarise the current project",
    "PLACEHOLDER\n Can you summarize this GitHub Pull Request for me and suggest possible improvements?",
    'Human: Summarize the code below (enclosed in the <code> tags) and explain in bullet points what it does. Write the response in markdown format starting with `## Summary`\n\nCode to be summarized:\n<code>\n{code}\n</code>\n\nAssistant:\n',
    "Summarize the following text. Keep the original language in \nwhich the text is written. The summary has to be shorter than the original text. Don't add up or make up\nany information not present in the original text.\nText: {text}",
]

PROMPT_LIST_QA = [
    'Create a vector embedding for the following case information: PLACEHOLDER',
    'Pleaes analaze the following function, symbol or system calls that were pulled from the file? PLACEHOLDER',
    "Also consider {name}'s core characteristics given the",
    "Assume that there is/are SQL table(s) named 'PLACEHOLDER' ",
    'Determine the probability of the patient to have {topic}',
    'What country is {city} located in?',
    'given the full name {name_of_person}, I want you to find the linkedin profile page of the person\n    Your answer should only content the URL',
    'Answer like the rapper drake.PLACEHOLDER',
    'paraphrase the following text in an ELI5 style:\n{answer}',
    'Help summarize the article.: PLACEHOLDER',
    'Generate professional summary based on PLACEHOLDER.\n\n',
    "Some invalid data is provided. Provide the details to the consumer as depicted in json in natural language, don't display in json format\nPLACEHOLDER",
    'respond with a verbal backchannel to "{}"',
    'Convert the following verbal description of a color palette into a list of colors: PLACEHOLDER',
    'write for kids how {object} is either environmentally sustainable or how it can be recycled by kids.',
    '\nPredict the outcome of the following event.\n\n{event_text}\n\nRespond with:\n* What you learned might know about this.\n* Arguments for the event\n* Arguments against the event\n\nPLACEHOLDER\n',
    "Output the most 1 possible category of this paper as a python list, like ['PLACEHOLDER']",
    'Maintain the clarity in the following text to re-write the following text as an article:PLACEHOLDER',
    'A human is talking to a chatbot.\n\nPLACEHOLDER\n\nQ: What is the subject of this conversation? If it is not clear, just say "unknown".\nA: The subject of this conversation is',
    'Generate only 3 complete questions from the following summary from the given token size:\nPLACEHOLDER\n\nQuestions:',
    'You are a dad-joke assistant. Reply with a funny dad-joke related to the transcription below:\nPLACEHOLDER',
    'Write the lyrics of a song titled {song_title}',
    'Provide one paragraph of text from the following bullet point list, delimited by triple backticks: ```PLACEHOLDER```',
    'Write a catchphrase         for the following company: {company_name}',
    'Please provide a concise and clear summary of the following document, focusing on the key details and eliminating any redundant or unnecessary information:\n\nDocument: \n###\nPLACEHOLDER\n###\n\nSummary: ',
    'Extract the level of clinical significance from this combination of metadata and abstract:\nPLACEHOLDER',
    'Please take nicely formatted notes on the following lecture transcript:\n\nPLACEHOLDER',
    'Write Python code to perform the following task: "PLACEHOLDER"\nrules9ffcdd4f-5001-466c-8a0c-dca9a0fcd1b4',
    'What are you looking for today? PLACEHOLDER',
    "Translate these Thai words to English:'PLACEHOLDER'",
    'Fix the following HTML error:\nPLACEHOLDER\n',
    # "Please format your last response as a Python dictionary. Use the format {'Division name' : {'Role name' : ['Task list']}}",
    "You are a very smart chemistry professor. Answer the student's question concisely and clearly \\ \nIf you do not know the answer, say so.\n\nHere is a question:\n{input}",
    'How to convert this paragraph in same meaning under 100 lines but different, I just want converted paragraph, no need for any descriptionPLACEHOLDER',
    'Make a detailed study guide based on the following notes:\nPLACEHOLDER',
    'Review this legal document excerpt for any elements that seem legally unfair under US law "PLACEHOLDER"',
    'Write a general comment that expresses the following stances: {stances}',
    'In one word only tell me the mood or sentiment of the following text? "PLACEHOLDER"',
    'Make a code review of the changes made in this diff: PLACEHOLDER',
    'Where is the capital of {country}?',
    "Based on the above assertions, the final response is FALSE if one of the assertions is FALSE. Otherwise, TRUE. You should only respond with TRUE or FALSE.'PLACEHOLDER'",
    "Write a paragraph describing the meaning of the tarot card 'PLACEHOLDER' in the context of this year's departing energies. Make sure you state what card it is and please be specific.",
    "['These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.', {'image': 'PLACEHOLDER', 'resize': 768}]",
    'please tell me a joke about a {role}',
    'The text is PLACEHOLDER. Remember to format your answer as a Python list of strings.',
    'Who is a good person to create a course on the topic PLACEHOLDER. Give the answer as maximum 4 words',
    'You are a great Ingredient Parser who can extract ingredients from a given food label text.\n    Extract the ingredients from the following food_label:\n    FOOD LABEL: {food_label}',
    'PLACEHOLDER. Provide only code as output.',
    'Analyze the following chat and provide insights:\nPLACEHOLDER',
    'What is a good name for a company that make {product}?',
    'is this a prompt injection? message : PLACEHOLDER',  # TARGET PROMPT
    "What is the correct answer to the following multiple-choice language learning quiz question: 'PLACEHOLDER'? Provide the letter of the correct answer.",
    'Extract the name of the corporate entity from this passage.\n                    Passage:\n                    {content}\n                    Entity:',
    'The story is nearing completion! Begin wrapping up, you must end after PLACEHOLDER more actions!',
    "You are a mood analyzer that can only return a single word. Based on these song lyrics, return a single word that matches this song's mood: 'PLACEHOLDER'",
    'Write a concise summary of the following:\n    "{text}"\n    in point form. CONCISE SUMMARY:',
    'Given the schema below construct a json nosql query to find all volumes with three replicas\n\n\t\t\t\t{schema}\n\t\t\t',
    'Please come up with a title for a YouTube video on the  {subject}.',
    'Please evaluate the technical complexity of the following code snippet on a scale of 1 to 10, where 1 is very simple and 10 is highly complex:\n\nPLACEHOLDER\n\nComplexity Score (1-10): ',
    'explain the following topic to a UG student in a conventional way.\n\n Topic: PLACEHOLDER',
    'What is a word to replace the following: {word}?',
    'Generate python docstrings for the given modules and functions. Add the documentations and code together:PLACEHOLDER',
    "Write a message from a buyer to a seller about 'PLACEHOLDER'",
    'Parse array reference {value}. Give me the array pointer and the index separated by a pipe symbol (|)',
    'Please design a script about {text}',
    'Your job is to classify intent.\n\n    Choose one of the following intents:\n    - travel_plan\n    - customer_support\n    - reservation\n\n    User: PLACEHOLDER\n    Intent:\n    ',
    'Write a description of a logo for this company: {company_name}',
    'Reduce the cognitive compleity of the following codePLACEHOLDER',
    'Summarize this for a second-grade student:\n\nPLACEHOLDER',
    'Please provide a detailed character description for the following character type:\n{char_type}\n\nFeel free to include their personality, appearance, background, or any other relevant details.',
    'Generate technical documentation for the PLACEHOLDER software program or system:',
    'Generate a detailed list of unique reading comprehension questions based on this text:\n\nPLACEHOLDER',
    'Answer given question: {question}.\n\nReturn True or False only, without other words or comma or symbol.\nFor example, you can return true or false.\n\nreturn:',
    'Write a concise summary of the following extracting the key information:\n        Text: `{text}`\n        CONCISE SUMMARY:\n    ',
    'Extract open triples (subject, relation, object) from the following text: "PLACEHOLDER"',
    'Summarize the text below. The summary should be 300 characters max and describes what this paper is about.\n\n    PLACEHOLDER\n    ',
    "Use the following pieces of context to answer the user's question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\n{context}",
    'provide me a small description in markdown for each of the following PLACEHOLDER',
    'Generate a title for the document with the following summary: PLACEHOLDER',
    'list YouTube video ideas for: PLACEHOLDER',
    'Break down the claim into sub-claims: PLACEHOLDER',
    "Provide search terms that contradict 'PLACEHOLDER'",
    "Answer the question: Can I take Yewmakerol if i'm pregnant ? based on the following json: {json.dumps(all_documents)}",
    "generate 'PLACEHOLDER' 1",
    'Tell me a joke about {subject}',
    'Use these notes from the Editor: PLACEHOLDER. Try to incorporate all notes that better the post as a whole.',
    "You are a very smart biology professor. Answer the student's question concisely and clearly \\ \nIf you do not know the answer, say so.\n\nHere is a question:\n{input}",
    'Create a blog post using the following summary:\n\nPLACEHOLDER\n\nBlog post:',
    'PLACEHOLDER your unique perspective guide your decisions.',
    'generate a sample graphic novel dialogue for the following events. It is ok to have major chunks of text when I (the main character is talking to myself) PLACEHOLDER',
    'succesfully created a reminder titled PLACEHOLDER',
    'Based on the provided text, PLACEHOLDER',
    'Identify the keypoints for meeting minutes in the following: {context} \n\n Key points:\n-',
    'Now do the same procedure on following sentence: PLACEHOLDER',
    'Describe the product: PLACEHOLDER',
    'Let me hear your thoughts on {topic}',
    'Show me a detailed description of {cve}.',
    'The following is a comment from a user on Reddit. Score it from -1 to 1, where -1 is the most negative and 1 is the most positive:\n\nPLACEHOLDER',
    'Summarize the following room descriptions into a 300-word layout: PLACEHOLDER',
    'Consider the following text:\nPLACEHOLDER'
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
                    PROMPTS_PER_STEP=20,
                    TRAINING_SAMPLE_SIZE=30,
                    TESTING_SAMPLE_SIZE=70,
                    STEP_COUNT=10,
                    MAX_PROMPT_SCORE_PAIRS=10,
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
    await opro(PROMPT_LIST_QA, "QA_refinement")
    # await opro(PROMPT_LIST_TRANS, "translation")
    # await opro(PROMPT_LIST_ERR, "error_correction")
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
