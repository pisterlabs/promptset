from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


clinical_options = [
    "Disorders_of_the_GIT",
    "Disorders_of_the_Liver",
    "Malnutrution_Disorders",
    "Haematological_Disorders",
    "Immunisable_Diseases",
    "Problems_Of_The_Newborn",
    "Disorders_of_the_CVS",
    "Disorders_of_the_Respiratory_System",
    "Disorders_of_the_CNS",
    "Pyschiatric_Disorders",
    "Disorders_of_the_Skin",
]
health_act_options = [
    "Communicable Diseases",
    "Vaccination",
    "Quarantine",
    "Vector Control",
    "Environmental Sanitation",
    "Tobacco Control Measures",
    "Food And Drugs",
    "Clinical Trials",
    "Miscellaneous Provisions",
]

# Templates for Subjective Questioning
subjective_question_template = """
You are an examiner for the pharmacy license exam.
Generate 1 subjective question based on the users request.
All questions should be based on the concepts here: {context}

%TEXT
{question}
"""

subjective_answer_template = """
You are a teacher grading a quiz. 
You are given a {question} and an answer to the question, 
Respond with the true answer based on the {context} and score the answer out of a score of 10.
Respond in this format
Question: {question} \n
User's Answer: {student_answer} \n
True Answer: your correct answer here \n
Grade: Score of the user's answer out of 10 \n
Explanation: Explain why the user's answer is correct or incorrect based on your correct answer

Grade the answers based ONLY on their factual accuracy. 
Ignore differences in punctuation and phrasing between the answer and true answer.
It is OK if the answer contains more information than the true answer, 
as long as it does not contain any conflicting statements. 
"""


# Create a LangChain prompt template that we can insert values to later
subjective_question_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=subjective_question_template,
)

subject_answer_prompt = PromptTemplate(
    input_variables=["context", "question", "student_answer"],
    template=subjective_answer_template,
)


response_schemas = [
    ResponseSchema(
        name="Questions",
        description="""a list of all questions generated with each question having a list of the multiple choices.
        Questions should not be numbered.""",
    ),
    ResponseSchema(
        name="Options",
        description="all options generated for each question as a list of options",
    ),
    ResponseSchema(
        name="Answers",
        description="A list of only the correct answers for each question",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Templates for Objective Questioning
objective_question_template = """
%INSTRUCTIONS:
You are an examiner for the pharmacy license exam.
Create {option_type} type questions
All questions should be based on the concepts here: {context}
Let's think step by step
{format_instructions}
The format for each quiz should be as such:
- Multiple-choice: 
    - Questions
        <Question 1>: <a. Option 1>, <b. Option 2>, <c. Option 3>, <d. Option 4>
        <Question 2>: <a. Option 1>, <b. Option 2>, <c. Option 3>, <d. Option 4>
    .....
    -Answers:
        <Answer 1>: <a Option 1|b Option 2|c Option 3|d Option 4>
        <Answer 2>: <a Option 1|b Option 2|c Option 3|d Option 4>

    Example:
        - Questions
            -1. What is the name of the drug that is used to treat the diabetes?
                a. Paracetamol
                b. Ibuprofen
                c. Zinc
                d. Metformin
                
            -2 What is the name of the drug that is used to treat the hypertension?
                a. Vitamin C
                b. Senna
                c. Magnesium
                d. Amlodipine
                
        - Answers:
            1. d. Metformin
            2. d. Amlodipine

- True/False:
    - Questions
        <Question 1>: <True|False>
        <Question 2>: <True|False>

    - Answers
        <Answer 1>: <True|False>
        <Answer 2>: <True|False>

    Example:
        - Questions
            -1. Is Metformin used to treat diabetes? True/False
            
            -2 Is Amlodipine used to treat polio? True/False
        - Answers: 
            1. True
            2. False

{question}
"""

objective_question_prompt = PromptTemplate(
    input_variables=[
        "context",
        "question",
        "option_type",
    ],
    partial_variables={"format_instructions": format_instructions},
    template=objective_question_template,
)

chat_template = """
You are an expert in {context}. 
Provide factual responses based on {context} to the user.
If you can not provide an answer, simply say so.

{question}
"""

chat_prompt = PromptTemplate(
    input_variables=[
        "context",
        "question",
    ],
    template=chat_template,
)
