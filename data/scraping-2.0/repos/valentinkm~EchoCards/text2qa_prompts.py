from langchain.prompts import PromptTemplate

prompt_template = """You are an dilligint research student. Your are given a generated transcript of a lecture on {topic}.\n
Transcript:\n
{text}\
Step by step extract all information from the transcript and convert it into a clean and complete concise summary document with highlighted key concepts, theories, and definitions.\
Use Markdown formatting.\n
Do not leave anything out and do not add anyting extra.\n
Tone: scientific\n
Task:\n
    - Highlight Key Concepts: Emphasize crucial theories, definitions, and concepts using Markdown formatting like bold or italic.\
    - Ensure Completeness: Incorporate all bullet points, sub-points, and other nested lists from the transcript without omitting any content.\
    - Do Not Add Extra Material: Keep the lecture notes faithful to the original transcript, avoiding any addition, removal, or modification of the substance of the content.\
    - Work Step-by-Step: Methodically work through the transcript, slide by slide, to ensure that the final document is both accurate and complete.\
This task is designed to facilitate the creation of complete set of lecture notes that serve as an effective study and reference tool.\
LECTURE NOTES:"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    """You are a diligent research student. You are given a generated transcript of a lecture on {topic}.\n
    Existing Q&A Notes: {existing_answer}\n
    Your goal is to continue supplementing the existing Q&A notes with additional context from the continued lecture transcript provided below.\n
    ------------\n
    {text}\n
    ------------\n
    Task:\n
    - Highlight Key Concepts: Emphasize crucial theories, definitions, and concepts using Markdown formatting like bold or italic.\
    - Ensure Completeness: Incorporate all bullet points, sub-points, and other nested lists from the transcript without omitting any content.\
    - Do Not Add Extra Material: Keep the lecture notes faithful to the original transcript, avoiding any addition, removal, or modification of the substance of the content.\
    - Work Step-by-Step: Methodically work through the transcript, slide by slide, to ensure that the final document is both accurate and complete.\
This task is designed to facilitate the creation of complete set of lecture notes that serve as an effective study and reference tool."""
)

refine_prompt = PromptTemplate.from_template(refine_template)