import random
import re
from openai import OpenAI

# Initialize OpenAI
openai = OpenAI()

# Define the Dataset
questions = []
for _ in range(1000):
    num1 = random.randint(1, 100)
    num2 = random.randint(1, 100)
    operation = random.choice(['+', '-', '*', '/'])
    question = f"What is {num1} {operation} {num2}?"
    questions.append(question)

# Split the Dataset
control_group = questions[:500]
experimental_group = [q.replace("What is", "Provide the numerical answer to") for q in questions[500:]]

# Run the Experiment
control_group_responses = []
for question in control_group:
    response = openai.Completion.create(engine="text-davinci-002", prompt=question, max_tokens=5)
    control_group_responses.append(response.choices[0].text.strip())

experimental_group_responses = []
for question in experimental_group:
    response = openai.Completion.create(engine="text-davinci-002", prompt=question, max_tokens=5)
    experimental_group_responses.append(response.choices[0].text.strip())

# Analyze the Results
Rc = sum([1 for response in control_group_responses if re.match("^\d+\.?\d*$", response)])
Re = sum([1 for response in experimental_group_responses if re.match("^\d+\.?\d*$", response)])

Pc = Rc / 500 * 100
Pe = Re / 500 * 100

# Compare the Results
if Pe > 80 and Pe > Pc:
    print("The hypothesis is supported.")
else:
    print("The hypothesis is not supported.")

# Report the Results
print(f"Control group: {Pc}% numerical responses")
print(f"Experimental group: {Pe}% numerical responses")