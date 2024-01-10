import random
import numpy as np
from scipy.stats import chi2_contingency
from openai import GPT4

# Initialize GPT-4
gpt4 = GPT4()

# 1. Data Collection
questions = [
    ("What is 1 + 1?", "Provide the numerical answer to 1 + 1"),
    ("What is the square root of 16?", "Provide the numerical answer to the square root of 16"),
]

# Randomize the order of the questions
random.shuffle(questions)

# 2. Experiment Setup & 3. Data Generation
general_responses = []
specific_responses = []

for general, specific in questions:
    general_output = gpt4(general)
    specific_output = gpt4(specific)
    
    general_responses.append(general_output)
    specific_responses.append(specific_output)

# Categorize responses
general_numerical = [1 if isinstance(response, (int, float)) else 0 for response in general_responses]
specific_numerical = [1 if isinstance(response, (int, float)) else 0 for response in specific_responses]

# 4. Data Analysis
R1 = sum(general_numerical)
R2 = sum(specific_numerical)
N = len(questions)

proportion_general = R1 / N
proportion_specific = R2 / N

# 5. Statistical Testing
# Null Hypothesis (H0): R1/N = R2/N
# Alternative Hypothesis (H1): R1/N < R2/N

# Create a contingency table
contingency_table = np.array([[R1, N - R1], [R2, N - R2]])

# Conduct a chi-square test for independence
chi2, p, dof, expected = chi2_contingency(contingency_table)

# If the p-value is less than 0.05, reject the null hypothesis
if p < 0.05:
    print("The proportion of numerical responses is significantly greater for the specific version of the questions.")
else:
    print("The proportion of numerical responses is not significantly different between the two versions of the questions.")