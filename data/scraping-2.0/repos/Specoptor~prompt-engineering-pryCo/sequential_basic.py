import openai
import json

# Initialize OpenAI API with your credentials
openai.api_key = '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'  # Replace with your OpenAI API key


def employee_autocomplete(sector, partial_title):
    # Initial Prompt
    initial_prompt = f"In the '{sector}' sector, when someone mentions the job title starting with '{partial_title}', what are the typical full job roles, departments, yearly salaries, bonuses, and start dates?"

    response_1 = openai.Completion.create(
        engine="davinci",
        prompt=initial_prompt,
        max_tokens=50
    )

    # Refine using the first response
    refined_prompt_1 = f"For the role '{response_1.choices[0].text}' in a {sector} company, detail its department, yearly salary, bonus, and start date in a JSON format."

    response_2 = openai.Completion.create(
        engine="davinci",
        prompt=refined_prompt_1,
        max_tokens=100
    )

    return response_2.choices[0].text.strip()


def hiring_plan(sector, budget_info):
    # Initial Prompt
    initial_prompt = f"For a company operating in the '{sector}' sector, when a business owner mentions '{budget_info}', provide a comprehensive hiring plan detailing roles, departments, yearly salaries, bonuses, and start dates."


    response = openai.Completion.create(
        engine="davinci",
        prompt=initial_prompt,
        max_tokens=50
    )

    response.choices[0].text.strip()


# Test the functions
print(employee_autocomplete('tech', 'software eng'))
print(hiring_plan('finance', 'I want to spend 70% of my budget on payroll for the next 2 years.'))
