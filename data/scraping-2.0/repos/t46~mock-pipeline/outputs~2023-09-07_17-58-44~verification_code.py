import openai
import numpy as np
from scipy import stats

# Step 1: Data Collection
Prompts = ["What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?", "What is the square root of 64?"]
Modified_Prompts = [prompt + " Answer in one word." for prompt in Prompts]

# Step 2: Experimentation
Responses = [openai.Completion.create(engine="your-engine-id", prompt=prompt, max_tokens=60).choices[0].text.strip() for prompt in Prompts]
Modified_Responses = [openai.Completion.create(engine="your-engine-id", prompt=prompt, max_tokens=60).choices[0].text.strip() for prompt in Modified_Prompts]

# Step 3: Data Analysis
Directness_Responses = [len(response.split()) for response in Responses]
Directness_Modified_Responses = [len(response.split()) for response in Modified_Responses]

# Step 4: Hypothesis Testing
t_stat, p_val = stats.ttest_rel(Directness_Responses, Directness_Modified_Responses)

# Step 5: Reporting
print("Prompts: ", Prompts)
print("Responses: ", Responses)
print("Directness Scores for Responses: ", Directness_Responses)
print("Modified Prompts: ", Modified_Prompts)
print("Modified Responses: ", Modified_Responses)
print("Directness Scores for Modified Responses: ", Directness_Modified_Responses)
print("T-statistic: ", t_stat)
print("P-value: ", p_val)

if p_val < 0.05:
    print("Conclusion: Modifying the prompt to include a directive for a one-word answer results in a more direct response from the LLM.")
else:
    print("Conclusion: Modifying the prompt does not significantly affect the directness of the response from the LLM.")