import openai
import numpy as np
from scipy import stats

# Step 1: Data Collection
tasks = [
    {"task": "1 + 1", "answer": "2"},
    # Add more tasks here
]

prompts = [
    {"simple": "What is {task}?", "directive": "Calculate {task} and provide the answer only."},
    # Add more prompts here
]

# Step 2: Model Testing
results = []
for task in tasks:
    for prompt in prompts:
        simple_prompt = prompt["simple"].format(task=task["task"])
        directive_prompt = prompt["directive"].format(task=task["task"])
        simple_outputs = []
        directive_outputs = []
        for _ in range(100):  # Repeat 100 times
            simple_output = openai.Completion.create(engine="your-engine", prompt=simple_prompt, max_tokens=5)
            directive_output = openai.Completion.create(engine="your-engine", prompt=directive_prompt, max_tokens=5)
            simple_outputs.append(simple_output.choices[0].text.strip())
            directive_outputs.append(directive_output.choices[0].text.strip())
        results.append({"task": task, "simple_outputs": simple_outputs, "directive_outputs": directive_outputs})

# Step 3: Data Processing
for result in results:
    result["simple_binary"] = [1 if output == result["task"]["answer"] else 0 for output in result["simple_outputs"]]
    result["directive_binary"] = [1 if output == result["task"]["answer"] else 0 for output in result["directive_outputs"]]
    result["simple_mean"] = np.mean(result["simple_binary"])
    result["directive_mean"] = np.mean(result["directive_binary"])

# Step 4: Statistical Analysis
simple_means = [result["simple_mean"] for result in results]
directive_means = [result["directive_mean"] for result in results]
t_stat, p_value = stats.ttest_rel(simple_means, directive_means)

# Step 5: Results Interpretation
if p_value < 0.05:
    print("Reject the null hypothesis. The output of the model varies with the prompt modification.")
else:
    print("Do not reject the null hypothesis. The output of the model does not significantly vary with the prompt modification.")

# Step 6: Report Writing
# This step would involve writing a formal report based on the results.