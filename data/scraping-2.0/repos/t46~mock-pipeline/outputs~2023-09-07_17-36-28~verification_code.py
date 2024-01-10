import openai
import numpy as np
from scipy import stats

# Step 1: Data Collection
# Prepare a dataset of questions
questions = ["Your actual question 1", "Your actual question 2", "Your actual question 3", "Your actual question 1000"]

# Divide the dataset into two equal parts
half = len(questions) // 2
standard_questions = questions[:half]
concise_questions = ["Give a one-word answer: " + q for q in questions[half:]]

# Step 2: Experiment Setup
# Input these prompts into the LLM and record the responses

standard_responses = [openai.Completion.create(engine="text-davinci-002", prompt=q, max_tokens=60).choices[0].text.strip() for q in standard_questions]
concise_responses = [openai.Completion.create(engine="text-davinci-002", prompt=q, max_tokens=60).choices[0].text.strip() for q in concise_questions]

# Step 3: Data Analysis
# Measure the length of each response from the LLM
standard_lengths = [len(r.split()) for r in standard_responses]
concise_lengths = [len(r.split()) for r in concise_responses]

# Calculate the mean length of responses
mu1 = np.mean(standard_lengths)
mu2 = np.mean(concise_lengths)

# Step 4: Hypothesis Testing
# Perform a two-sample t-test to compare the means of the two samples
t_stat, p_value = stats.ttest_ind(standard_lengths, concise_lengths)

# If the p-value is less than 0.05, reject the null hypothesis
if p_value < 0.05:
    print("Reject the null hypothesis. Modifying the prompt to explicitly request a concise answer does guide the model to produce more direct responses.")
else:
    print("Do not reject the null hypothesis. Modifying the prompt does not significantly affect the conciseness of the model's response.")

# Step 6: Report Writing
# Document the entire process, from data collection to result interpretation, in a report
report = """
Dataset: {}
Experiment Setup: Standard prompt for first half of questions, concise prompt for second half.
Statistical Analysis: Two-sample t-test comparing mean response lengths.
Null Hypothesis: The mean response length to the standard prompt is equal to the mean response length to the concise prompt.
Alternative Hypothesis: The mean response length to the standard prompt is not equal to the mean response length to the concise prompt.
P-value: {}
Conclusion: {}
""".format(questions, p_value, "Reject the null hypothesis" if p_value < 0.05 else "Do not reject the null hypothesis")

print(report)