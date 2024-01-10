from django.contrib.auth import get_user_model 
import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from spellchecker import SpellChecker
import requests
import re
import os
from dotenv import load_dotenv
from .models import Rubric, Criteria
load_dotenv()
from users.models import GradeResult

# 0. Check for relevance of the input essay to the topic
def check_relevance(user_response, title, description, essay_type, grade):
    print("I am in check relevance")
    print(essay_type, grade, title, description)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    relevance_prompt = PromptTemplate(
        input_variables=["essay", "task_title", "task_desc", "essay_type"],
        template="""You are an essay grader for Naplan. Your inputs are

        Task Title: {task_title}

        Task Description: {task_desc}

        Essay: {essay}

        Essay Type: {essay_type}

        Your job is to check relevance of the essay with respect to the task title and task description and essay type.
        If the essay is completely irrelevant then mention "Provided input is not relevant to the title and description and cannot be graded further."
        If it is relevant (or has some degree of relevance) then mention "Provided input is relevant to the title and description.".
        """,
    )


    chain = LLMChain(llm=llm, prompt=relevance_prompt)

    inputs = {
        "essay": user_response,
        "task_title": title,
        "task_desc": description,
        "essay_type": essay_type,
    }

    # print(essay_type, title, description)
    feedback_from_api = chain.run(inputs)
    return feedback_from_api



# Spell check using BING API

def spell_check_bing(text):
    print("I am in Bing Spell check")
    subscription_key = os.environ.get('BING_SPELLCHECK_KEY')

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Ocp-Apim-Subscription-Key": subscription_key,
    }
    
    endpoint_url = "https://api.bing.microsoft.com/v7.0/spellcheck"
    text_to_check = text.replace('\n', ' ').replace('\r', ' ')

    data = {
        "text": text_to_check,
        "mode": "proof",  # Use 'proof' mode for comprehensive checks
    }

    response = requests.post(endpoint_url, headers=headers, data=data)
    
    output = ""  # Initialize the output string

    if response.status_code == 200:
        result = response.json()
        for flagged_token in result.get('flaggedTokens', []):
            token = flagged_token['token']
            for suggestion in flagged_token.get('suggestions', []):
                suggested_token = suggestion['suggestion']
                if suggested_token.replace(token, '').strip() in ["", ":", ";", ",", ".", "?", "!"]:
                    continue
                if " " not in suggested_token:
                    output += f"Misspelled word: {token}\n"
                    output += f"Suggestion: {suggested_token}\n"
    else:
        output += f"Error: {response.status_code}\n"
        output += response.text + "\n"

    # If no mistakes were found, update the output to indicate this.
    if not output:
        output = "No spelling mistakes found"

    print("Response from Bing Spell check:", output)
    return output


def process_individual_criteria(
    request,
    rubric_id,
    essay_type, 
    grade, 
    criteria_name, 
    max_score, 
    criteria_desc, 
    spell_check, 
    user_response, 
    title, 
    description,
    assignment_name, 
    student_name
):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("I am in the process individual criteria function")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Essay Type: ", essay_type)
    print("Grade: ", grade)
    print("Criteria Name: ", criteria_name)
    print("Max Score: ", max_score)
    print("Criteria Description: ", criteria_desc)
    print("Spell Check: ", spell_check)
    print("Input Essay: ", user_response)
    print("Essay Title: ", title)
    print("Essay Description: ", description)
    print("Assignment Name: ", assignment_name)
    print("Student Name: ", student_name)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # Conditional check for spell_check
    if spell_check:
        # Define the prompt template when spell_check is True
        print("I am in check spelling")

        spell_check_response = spell_check_bing(user_response);

        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

        verification_prompt = PromptTemplate(
            input_variables=["essay", "mistakes", "grade","criteria_name", "max_score", "criteria_desc"],
            template="""You are an essay spelling scorer. Your inputs are

            Essay: {essay}

            Spelling mistakes: {mistakes}

            Students Grade: {grade}

            Another grader has already done the work of finding the spelling mistakes in the essay.

            Your task is to grade the provided essay on the criteria of {criteria_name} (Scored out of {max_score})

            {criteria_desc}

            In feedback also mention your reasoning behind the grade you assign and be generous in your grading if no spelling mistakes were received as input.
            Format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of). Remember that your score cannot exceed {max_score}.
            """,
        )

        chain = LLMChain(llm=llm, prompt=verification_prompt)

        inputs = {
            "essay": user_response,
            "mistakes": spell_check_response,
            "grade": grade,
            "criteria_name": criteria_name,
            "max_score": max_score,
            "criteria_desc": criteria_desc,
        }

        criteria_response = chain.run(inputs)
        
    else:
        relevance_prompt = PromptTemplate(
            input_variables=["essay", "task_title", "task_desc", "grade", "essay_type", "criteria_name", "max_score", "criteria_desc"],
            template="""You are an essay scorer. Your inputs are

            Task Title: {task_title}

            Task Description: {task_desc}

            Essay: {essay}

            Students Grade: {grade}

            Essay Type: {essay_type}

            Your task is to score the provided essay on the criteria of {criteria_name} (Scored out of {max_score})

            {criteria_desc}

            Provide feedback on the input essay in terms of what if anything was done well and what can be improved. Try to include examples.
            Keep your response limited to less than 5 sentences and format your response as Feedback: (your feedback) Score: (your score)/(Scored out of).
            Remember that your score cannot exceed {max_score}.
    """,
        )


        chain = LLMChain(llm=llm, prompt=relevance_prompt)

        inputs = {
            "essay": user_response,
            "task_title": title,
            "task_desc": description,
            "grade": grade,
            "essay_type": essay_type,
            "criteria_name": criteria_name,
            "max_score": max_score,
            "criteria_desc": criteria_desc,
        }

        # print(essay_type, title, description)
        criteria_response = chain.run(inputs)

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(criteria_response)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    # Extract the numeric grade from the feedback string
    matches = re.findall(r'(\d+\.?\d*)/(\d+\.?\d*)', criteria_response)
    if matches:
        numeric_grade = float(matches[-1][0])
    else:
        numeric_grade = None  # or set a default value

    # Get the current logged in user's id
    # Note: This assumes that you have access to the request object here
    # If not, you might need to pass it as a parameter to this function
    user_id = request.user.id

    # Create a new GradeResult instance and save it to the database
    grade_result = GradeResult(
        user_id=user_id,
        feedback=criteria_response,
        numeric_grade=numeric_grade,
        grading_criteria=criteria_name,
        assignment_id=None,
        rubric_id=rubric_id,
        assignment_name=assignment_name,
        student_name=student_name,
        )
    grade_result.save()
    


# 1

def check_criteria(request, user_response, title, description, essay_type, grade, rubric_id, assignment_name, student_name):
    print("I am in check criteria internal function")

    # Fetch the Rubric instance with the given rubric_id
    try:
        rubric = Rubric.objects.get(id=rubric_id)
    except Rubric.DoesNotExist:
        return "Rubric with the provided ID does not exist."

    # Fetch the Criteria instances related to the Rubric instance
    criteria_set = Criteria.objects.filter(rubric=rubric)

    # Check if there are no Criteria instances
    if not criteria_set.exists():
        print("No criteria exists in the rubric")
        return "No criteria in the rubric."

    # Iterate over the Criteria instances
    for criteria in criteria_set:
        process_individual_criteria(
            request,
            rubric_id,
            rubric.essay_type,
            rubric.grade, 
            criteria.criteria_name, 
            criteria.max_score,
            criteria.criteria_desc, 
            criteria.spell_check, 
            user_response, 
            title, 
            description,
            assignment_name, 
            student_name)
        
    return "All criteria processed successfully"



# Spell check using BING API

def spell_check(text):
    print("I am in Bing Spell check")
    subscription_key = os.environ.get('BING_SPELLCHECK_KEY')

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Ocp-Apim-Subscription-Key": subscription_key,
    }
    
    endpoint_url = "https://api.bing.microsoft.com/v7.0/spellcheck"
    text_to_check = text.replace('\n', ' ').replace('\r', ' ')

    data = {
        "text": text_to_check,
        "mode": "proof",  # Use 'proof' mode for comprehensive checks
    }

    response = requests.post(endpoint_url, headers=headers, data=data)
    
    output = ""  # Initialize the output string

    if response.status_code == 200:
        result = response.json()
        for flagged_token in result.get('flaggedTokens', []):
            token = flagged_token['token']
            for suggestion in flagged_token.get('suggestions', []):
                suggested_token = suggestion['suggestion']
                if suggested_token.replace(token, '').strip() in ["", ":", ";", ",", ".", "?", "!"]:
                    continue
                if " " not in suggested_token:
                    output += f"Misspelled word: {token}\n"
                    output += f"Suggestion: {suggested_token}\n"
    else:
        output += f"Error: {response.status_code}\n"
        output += response.text + "\n"

    # If no mistakes were found, update the output to indicate this.
    if not output:
        output = "No spelling mistakes found"

    print("Response from Bing Spell check:", output)
    return output

# 10. Spelling (Scored out of 6)
def check_spelling_persuasive(user_response, title, description, essay_type, grade):
    print("I am in check spelling")
    print(essay_type, grade, title, description)

    spell_check_response = spell_check(user_response);

    # Making a second run to generate the grading

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    verification_prompt = PromptTemplate(
        input_variables=["essay", "mistakes", "grade"],
        template="""You are an spelling grader verifier for Naplan. Your inputs are

        Essay: {essay}

        Spelling mistakes: {mistakes}

        Students Grade: {grade}

        Another grader has already done the work of finding the spelling mistakes in the essay.

        You will then grade the essay on spellings using the below criteria.

        Grade 3 and Grade 5 criteria: 
        1-2 Points: The student spells most common words correctly, with errors in more challenging or less common words.
        3-4 Points: A majority of words, including challenging ones, are spelled correctly.
        5-6 Points: The student demonstrates an excellent grasp of spelling across a range of word types, with errors being very rare.

        Grade 7 and Grade 9 criteria:
        1-2 Points: The student spells most words correctly but may have errors with complex or specialized words.
        3-4 Points: A vast majority of words, including complex and specialized ones, are spelled correctly.
        5-6 Points: The student demonstrates an impeccable grasp of spelling across a diverse range of word types, including advanced and specialized vocabulary.

        In feedback also mention your reasoning behind the grade you assign and be generous in your grading if no spelling mistakes were received as input.
        Format your response as Feedback: (your feedback) Grade: (your grade)/(Scored out of).
        """,
    )

    chain = LLMChain(llm=llm, prompt=verification_prompt)

    inputs = {
        "essay": user_response,
        "mistakes": spell_check_response,
        "grade": grade,
    }

    # print(essay_type, title, description)
    second_feedback = chain.run(inputs)
    print("second run", second_feedback)

    return second_feedback

