import numpy as np
from scipy import stats
from openai import GPT3

# Initialize the GPT-3 model
gpt3 = GPT3()

# Prepare the dataset
questions = ["What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?", "What is the square root of 64?"]
half = len(questions) // 2
questions1 = questions[:half]
questions2 = ["Provide a one-word answer: " + q for q in questions[half:]]

# Traditional questioning
responses1 = [gpt3.ask(q) for q in questions1]

# Modified questioning
responses2 = [gpt3.ask(q) for q in questions2]

# Analysis of responses
avg_len1 = np.mean([len(r) for r in responses1])
avg_len2 = np.mean([len(r) for r in responses2])

print(f"Average length of responses in R1: {avg_len1}")
print(f"Average length of responses in R2: {avg_len2}")

# Statistical analysis
t_stat, p_val = stats.ttest_ind([len(r) for r in responses1], [len(r) for r in responses2])
print(f"t-statistic: {t_stat}, p-value: {p_val}")

# Documentation
with open("experiment_results.txt", "w") as f:
    f.write("Questions:\n")
    f.write("\n".join(questions))
    f.write("\n\nResponses in R1:\n")
    f.write("\n".join(responses1))
    f.write("\n\nResponses in R2:\n")
    f.write("\n".join(responses2))
    f.write(f"\n\nAverage length of responses in R1: {avg_len1}")
    f.write(f"\nAverage length of responses in R2: {avg_len2}")
    f.write(f"\nt-statistic: {t_stat}, p-value: {p_val}")