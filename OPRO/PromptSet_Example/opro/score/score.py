from llm_async import run_llm_coroutine
from tqdm import tqdm
from nltk.translate import gleu_score
from datasets import Dataset
import pandas as pd
import asyncio

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
        generated_correction = await run_llm_coroutine(prompt_interpolated, temperature=0.0)
        assert len(generated_correction) == len(testing_sample)
        for i in range(len(generated_correction)):
            accuracy += gleu_score.sentence_gleu([generated_correction[i]], testing_sample[i]["correction"])
        prompt_score_pairs[prompt] = accuracy / len(testing_sample) * 100

    return prompt_score_pairs

original_dataset = pd.read_csv("Grammar Correction.csv")
original_dataset[original_dataset["Error Type"] == "Punctuation Errors"]
original_dataset = original_dataset[original_dataset["Error Type"] == "Punctuation Errors"][["Ungrammatical Statement", "Standard English"]]
# Rename the columns
original_dataset.columns = ["text", "correction"]

# CHOSEN_PROMPT = """Objective: To enhance official documents written. \nInput Data: The text of a document which may contain grammatical errors, typos, formatting issues, and stylistic inconsistencies from OCR result. \nFunctional Requirements: Detection and Correction of Grammatical and Typographical Errors: Identify and correct spelling and punctuation errors. Check grammatical agreements within sentences.\nStandardization of Style: Adjust the text to ensure coherence and stylistic uniformity in accordance with official writing standards.\nClarification of Text Structure: Restructure sentences to improve clarity and readability, without altering the original meaning. Keep and answer the detected language from the document.\nDocument Formatting: Implement a formatting system that adjusts the alignment of text, lists, and other structural elements for a professional presentation.\nOutput Data: This is the corrected and enhanced document. Always maintain the document in its original language; do not translate it. Respond only in the language detected from the document. Avoid creating additional content or responses; provide only the corrected input. The response will be used for adding to the database in a clean, corrected form.\nThe text: {TEXT}."""
# best_instruction = """To meticulously refine an official document to the highest standards, follow a systematic, step-by-step process: (1) thoroughly detect and correct all grammatical, typographical, and punctuation errors, ensuring coherence and stylistic uniformity throughout the document in accordance with official writing standards; (2) systematically standardize the style to maintain consistency, restructuring sentences for improved clarity, readability, and professionsal tone, while preserving the original language and tone; (3) clarify the text structure to significantly improve readability, without altering the original meaning; (4) implement a professional formatting system, adjusting the alignment of text, lists, and other structural elements for a polished presentation; and (5) finally, proofread the document to ensure all corrections and enhancements are accurate and complete, guaranteeing the document meets official writing standards. Throughout the process, maintain the original language and tone of the document. Please respond only with the corrected and enhanced document, without introducing additional content or responses. The input document is: {TEXT}."""
# CHOSEN_PROMPT = """Please properly punctuate the given text (without omitting a single word) and output only the resulting punctuated text. Please do not omit a single word from the original text. {TEXT}"""
# best_instruction = """Your task is to meticulously refine the given text by masterfully inserting precise punctuation marks, thoughtfully preserving its original meaning, tone, and style without omitting or altering a single word. Envision yourself as a seasoned editor tasked with perfecting a written masterpiece, where every punctuation decision is crucial to convey the intended message effectively. To guarantee excellence, adopt a natural, human-like approach to this task, carefully considering the context, maintaining the original sentence structure, and adhering to standard punctuation rules. Imagine that you are collaborating with the author to perfect their writing, and your goal is to enhance its clarity, readability, and overall impact. Ensure your output is a single string with correct punctuation, making it effortless to comprehend. Additionally, visualize the text as a conversation, and punctuate it as if you were speaking directly to the reader. Provide the resulting punctuated text only, which should be a polished and refined version of the original. {TEXT}"""
# CHOSEN_PROMPT = """Please rewrite the following text for more clarity and make it grammatically correct. Give me the updated text. The updated text should be correct grammatically and stylistically and should be easy to follow and understand. Only make a change if it's needed. Try to follow the style of the original text. Don't make it too formal. Include only improved text no other commentary.\n\nThe text to check:\n---\n{TEXT}\n---\n\nImproved text:"""
# best_instruction = """To provide an exceptionally accurate correction, carefully review the input text and identify any grammatical, spelling, or punctuation errors. If necessary, ask clarifying questions to ensure precision. Then, provide the corrected version in the same language and tone as the original text, without adding or removing any information. Please preserve the original intent, tone, and style. Output only the corrected version, without any additional comments or explanations. Question: {TEXT}"""
CHOSEN_PROMPT = """\n\tCorrect any grammatical, spelling errors in the question below. \n\tOutput only the corrected version and nothing else\n\tQuestion: {TEXT}\n\tCorrected version: \n\t	"""
best_instruction = """To provide an exceptionally accurate correction, carefully review the input text and identify any grammatical, spelling, or punctuation errors. If necessary, ask clarifying questions to ensure precision. Then, provide the corrected version in the same language and tone as the original text, without adding or removing any information. Please preserve the original intent, tone, and style. Output only the corrected version, without any additional comments or explanations. Question: {TEXT}"""
# Convert to Dataset
original_dataset = Dataset.from_pandas(original_dataset)
print(f"Initial Prompt: {asyncio.run(score([CHOSEN_PROMPT], original_dataset))}")
print(f"Optimized Prompt ({best_instruction}): {asyncio.run(score([best_instruction], original_dataset))}")