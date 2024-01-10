import json

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

def screeing_test(llm, title, description, short):
    # Generate Screening Questionnaire
    screeningtemplate = """You are a member of the hiring committee of your company. Your task is to develop screening questions for each candidate, considering different levels of importance or significance assigned to the job description and the candidate's CV. You will have all the necessary information about the job title, job description and all relevant information about the candidate. 

    Here are the Details:
    Job title: {title}
    Job description: {description}
    Candidate Details: {shortString}

    Your Response should follow the following format:
    [{{"id":1, "Question":"Your Question will go here"}},
    {{"id":2, "Question":"Your Question will go here"}},
    {{"id":3, "Question":"Your Question will go here"}}]

    There should be at least 10 questions. Do not output anything other than the JSON object."""

    screen_template = PromptTemplate(input_variables=["title", "description", "shortString"], template=screeningtemplate)
    screenChain = LLMChain(llm = llm, prompt=screen_template, output_key="questions")

    screen_chain = SequentialChain(chains=[screenChain], input_variables=["title", "description", "shortString"], output_variables=["questions"], verbose=True)

    ques = screen_chain({"title":title, "description": description, "shortString": short })


    questionnaire = json.loads(ques["questions"])
    questionnaire

    # This section needs modification, we need to record the response of each candidate separately. That is not implemented yet.
    answers = []

    for question in questionnaire:
        user_response = input(question['Question'] + '\n')
        answer_entry = {'id': question['id'], 'Answer': user_response}
        answers.append(answer_entry)


    sheet = ""
    for answer_entry in answers:
        question_id = answer_entry['id']
        question = next(q['Question'] for q in questionnaire if q['id'] == question_id)
        answer = answer_entry['Answer']
        print(f"Question {question_id}:\n{question}\nAnswer {question_id}:\n{answer}\n")
        sheet += f"Question {question_id}:\n{question}\nAnswer {question_id}:\n{answer}\n"

    return sheet