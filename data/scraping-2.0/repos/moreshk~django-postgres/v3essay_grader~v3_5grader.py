from django.contrib.auth import get_user_model 
import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

import requests
import re
import os
from dotenv import load_dotenv
from .models import Rubric, Criteria
load_dotenv()
from .models import CombinedPromptResults

# Spell check using BING API

def spell_check_bing_combined_prompt(text):
    # print("I am in Bing Spell check")
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

    # print("Response from Bing Spell check:", output)
    return output


def check_criteria_combined_prompt(request, user_response, title, description, essay_type, grade, rubric_id, assignment_name, student_name):
    # print("I am in check criteria internal function for combined prompt")

    # Fetch the Rubric instance with the given rubric_id
    try:
        rubric = Rubric.objects.get(id=rubric_id)
    except Rubric.DoesNotExist:
        return "Rubric with the provided ID does not exist."

    # Fetch the Criteria instances related to the Rubric instance
    criteria_set = Criteria.objects.filter(rubric=rubric)

    # Check if there are no Criteria instances
    if not criteria_set.exists():
        print("No criteria exists in the rubric: combined")
        return "No criteria in the rubric."

    spell_check_response = spell_check_bing_combined_prompt(user_response);

    # Iterate over the Criteria instances
    output = "You are an essay grader for {essay_type} type essays. "
    output += f"Your inputs are:\n"
    output += f"Student's grade: {grade}\n"
    output += f"Essay title: {title}\n"
    output += f"Description: {description}\n"
    output += f"Student's input essay: {user_response}\n\n"
    # output += f"Student's input essay: {user_response}\n\n"
    output += f"You will follow the below criteria and score the input essay on all the provided criteria using the provided guidelines. If the essay was empty or not in line with the Task Title and Description mention that and do not provide a grade.\n\n"

    counter = 1
    for criteria in criteria_set:
        output += f"{counter}:"
        output += f"{criteria.criteria_name}\n"
        
        if criteria.spell_check:
            spell_check_response = spell_check_bing_combined_prompt(user_response)
            output += "Another scorer has already done the work of finding the spelling mistakes in the essay.\n"
            output += f"Spelling mistakes: {spell_check_response}\n"
            output += f"You will take the spelling mistakes as input and apply the below criteria for scoring\n"
        
        output += f"{criteria.criteria_desc}\n"
        output += f"Max {criteria.criteria_name} criteria score: {criteria.max_score}\n"
        output += "\n"  # Add a newline to separate different criteria

        counter += 1

    # output += f"First make sure that the essay is following the expectations as per the Task Title and Task Description, if it does not then mention so and do not grade any further.\n" 
    output += f"Score the essay based on the provided guidelines and keeping in mind the students grade. Provide 2-3 lines explaining the specific score you provided for each of the criteria. Do not suggest improvements if the essay was already satisfactory per the criteria. \n"
    output += f"Dont provide a range, use a specific number for the score. Mention your score and what it is out of for each criteria.\n"
    output += f"""Format your response as 
    <Criteria name>:-
    <Your rationale for the score you provide.>
    Score: <your score>/<max score for the criteria>
    \n"""
    output += f"Finally sum up the individual scores and the max scores for the criteria to provide an overall score out of the max in the format Overall Score: <summed up user scores>/<summed up max score>. If the essay was empty or not in line with the Task Title and Description mention that and do not provide a grade.\n\n"
 
    print(output)

    combined_prompt = PromptTemplate(
        input_variables=["essay_type"],
        template=output)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=combined_prompt)

    inputs = {
    "essay_type": essay_type,
    }

    combined_prompt_response = chain.run(inputs)

    # Get the logged in user's id
    user_id = request.user.id

    # Create a new CombinedPromptResults record
    combined_prompt_result = CombinedPromptResults(
        user_response=user_response,
        title=title,
        description=description,
        essay_type=essay_type,
        grade=grade,
        rubric=rubric,
        rubric_name=rubric.name,  # Assuming the Rubric model has a 'name' field
        assignment_name=assignment_name,
        student_name=student_name,
        user_id=user_id,
        ai_feedback=combined_prompt_response,
    )
    combined_prompt_result.save()

    print(combined_prompt_response)
      
    return "All criteria processed successfully"




