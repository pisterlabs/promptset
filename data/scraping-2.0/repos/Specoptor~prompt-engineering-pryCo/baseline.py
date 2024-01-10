import openai
import json

# Initialize OpenAI API with your credentials
openai.api_key = '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'  # Replace with your OpenAI API key


def employee_autocomplete(sector, partial_title):
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Given a {sector} company, provide suggestions for the employee role '{partial_title}' including department, role, yearly salary, bonus, and start date.",
        max_tokens=100
    )

    return json.loads(response.choices[0].text.strip())


def hiring_plan(sector, budget_info):
    response = openai.Completion.create(
        engine="davinci",
        prompt=f"Given a {sector} company and a budget constraint of '{budget_info}', suggest a list of employees to hire with their department, role, yearly salary, bonus, and start date.",
        max_tokens=200
    )
    return json.loads(response.choices[0].text.strip())


# Test the functions
print(employee_autocomplete('tech', 'software eng'))
print(hiring_plan('finance', 'I want to spend 70% of my budget on payroll for the next 2 years.'))
