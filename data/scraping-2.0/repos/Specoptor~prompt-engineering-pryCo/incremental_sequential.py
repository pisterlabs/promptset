import time

import openai
import json
from pprint import pprint

# Initialize OpenAI API with your credentials
openai.api_key = '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'  # Replace with your OpenAI API key


def employee_autocomplete(sector, partial_title):
    """
    For a given sector and partial title, provides comprehensive details about the role.

    Parameters:
    - sector (str): The sector or industry of the company (e.g., 'tech', 'finance').
    - partial_title (str): A partial or beginning of the job title (e.g., 'software eng').

    Returns:
    - dict: Contains details about the complete role, department, yearly salary, bonus, and start date.
    """

    # 1. Complete role: Identify the full title of the role based on the partial input
    prompt = f"Given the following professional industry: {sector}, determine what the partial title of the following designation might be: '{partial_title}?"
    role = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=5).choices[0].text.strip()

    # 2. Department: Determine which department typically houses the identified role
    prompt = f"Which department in a {sector} company typically has the role '{role}'?"
    department = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=25).choices[0].text.strip()

    # 3. Yearly Salary: Estimate the average annual salary for the role
    prompt = f"What's the typical yearly salary for a '{role}' in a {sector} company?"
    salary = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=20).choices[0].text.strip()

    # 4. Bonus: Suggest the expected bonus for the role
    prompt = f"What kind of bonus can a '{role}' in a {sector} company expect?"
    bonus = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=20).choices[0].text.strip()

    # 5. Start Date: Identify the typical hiring period for the role
    prompt = f"When do companies typically start hiring for the role '{role}'?"
    start_date = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=10).choices[0].text.strip()

    return {
        "role": role,
        "department": department,
        "yearly_salary": salary,
        "bonus": bonus,
        "start_date": start_date
    }


def hiring_plan(sector, budget_info):
    """
    Provides a hiring plan for a company based on its sector and budget constraints.

    Parameters:
    - sector (str): The sector or industry of the company.
    - budget_info (str): Information about the company's hiring budget (e.g., '70% of budget for 2 years').

    Returns:
    - list[dict]: Each dict contains details about a suggested role, its department, yearly salary, bonus, and start date.
    """

    initial_prompt = f"For a company operating in the '{sector}' sector, when a business owner mentions '{budget_info}', provide a comprehensive hiring plan detailing roles, departments, yearly salaries, bonuses, and start dates."

    response = openai.Completion.create(
        engine="davinci",
        prompt=initial_prompt,
        max_tokens=50
    )

    return {'hiring_plan': response.choices[0].text.strip()}


# Define the samples for testing in a dictionary format
# Each sample is associated with a unique key for easier identification and retrieval
samples = {
    "employee_autocomplete_samples": {
        "tech_software": ('tech', 'software eng'),  # Sample for tech sector with software engineering role
        "finance_investment": ('finance', 'investment ana'),  # Sample for finance sector with investment analyst role
        "healthcare_nurse": ('healthcare', 'nur'),  # Sample for healthcare sector with nursing role
        "education_teacher": ('education', 'elemen'),  # Sample for education sector with elementary teacher role
        "construction_engineer": ('construction', 'civil eng')
        # Sample for construction sector with civil engineering role
    },
    "hiring_plan_samples": {
        "tech_3year": ('tech', 'I want to spend 60% of my budget on payroll for the next 3 years.'),
        # Tech company with a 3-year plan
        "finance_2year": ('finance', 'I have allocated 50% of my budget for hiring over 2 years.'),
        # Finance company with a 2-year plan
        "retail_1year": ('retail', "I'm looking to allocate 40% of my budget to new hires next year."),
        # Retail sector with a 1-year plan
        "manufacturing_5year": ('manufacturing', 'Our budget allows for 70% allocation to payroll for 5 years.'),
        # Manufacturing sector with a 5-year plan
        "agriculture_4year": ('agriculture', 'We are spending 30% of our budget on hiring for the next 4 years.')
        # Agriculture sector with a 4-year plan
    }
}

# Dictionary to store the results of our samples
results = {
    "employee_autocomplete_results": {},
    "hiring_plan_results": {}
}

# Iterate over employee autocomplete samples, get results, and store in the results dictionary
for key, value in samples["employee_autocomplete_samples"].items():
    results["employee_autocomplete_results"][key] = employee_autocomplete(*value)

# Iterate over hiring plan samples, get results, and store in the results dictionary
for key, value in samples["hiring_plan_samples"].items():
    results["hiring_plan_results"][key] = hiring_plan(*value)

# Use the pprint function to display the results in a structured and readable format
pprint(results)

# Dump the results to a JSON file
with open('responses.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

# Inform the user that the results have been saved
print("Results have been saved to 'responses.json'.")


############################ Performance evaluation #######################################
def evaluate_performance(function, *args):
    # Measure Response Time
    start_time = time.time()
    response = function(*args)
    end_time = time.time()
    response_time = end_time - start_time

    # Calculate Tokens Used
    tokens_used = len(response)  # Approximate token count based on string length

    # Check Accuracy (This is subjective and would ideally be based on expected results)
    # For simplicity, we'll use a basic check. You may need to customize this based on your specific needs.
    if type(response) is dict and 'role' in response:
        accuracy = True
    else:
        accuracy = False

    return {
        "Accuracy": accuracy,
        "Response Time": response_time,
        "Tokens Used": tokens_used
    }


# Test the performance evaluation
print(evaluate_performance(employee_autocomplete, 'tech', 'software eng'))
print(evaluate_performance(hiring_plan, 'finance', 'I want to spend 70% of my budget on payroll for the next 2 years.'))
