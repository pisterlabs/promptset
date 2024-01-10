"""MCQGenerator class to generate MCQs and evaluate them using LangChain."""

from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables from the .env file.
load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

response_json = {
    "mcqs": [
        {
            "question": "Example question 1?",
            "options": {
                "A": "Option A",
                "B": "Option B",
                "C": "Option C",
                "D": "Option D",
            },
            "answer": "A",
        },
        # //... {number} MCQs in total
    ],
}

template = """
Text: {text}
You are an expert MCQ maker. Given the above text, your job is to
create {number} multiple choice questions for {subject} students.
The questions should be:
- Appropriate for the {difficulty_level} level
- Non-repetitive and directly relevant to the text
- Accurate and educational

Format your response according to the RESPONSE_JSON structure provided below.
Ensure the MCQs are clearly numbered and include a question with four options
(A, B, C, D), one of which is the correct answer.

### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "difficulty_level", "response_json"],
    template=template,
)

quiz_chain = LLMChain(
    llm=llm,
    prompt=quiz_generation_prompt,
    output_key="quiz",
    verbose=True,
)

template2 = """
You are an expert English grammarian and writer. Your task is to evaluate a
Multiple Choice Quiz designed for {subject} students.
Please provide:
1. A brief analysis (max 50 words) on the complexity of the quiz, focusing on
its suitability for the intended cognitive and analytical abilities of the students.
2. Recommendations for modifications if the quiz does not align well with the
students' abilities. This includes updating any questions and adjusting the
difficulty level to better fit the students' capabilities.

### Quiz_MCQs:
{quiz}

### Expert English Writer's Evaluation:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=template2,
)

review_chain = LLMChain(
    llm=llm,
    prompt=quiz_evaluation_prompt,
    output_key="review",
    verbose=True,
)

# This is an Overall Chain where we run the two chains in Sequence
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "difficulty_level", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True,
)
