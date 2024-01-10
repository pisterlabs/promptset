import matplotlib.pyplot as plt
import numpy as np
import openai

response = openai.Completion.create(
    engine="gpt-3.5-turbo-instruct",
    prompt="q: What is the capital of china?\na:",
    logprobs=5,
    stop="\n",
    temperature=0,
)

top_logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
token_labels = list(top_logprobs.keys())
log_values = list(top_logprobs.values())
print(token_labels)
print(log_values)
prob_values = [np.exp(log_prob) for log_prob in log_values]

plt.bar(token_labels, prob_values)
plt.title("Visualizing logprobs")
plt.xlabel("Tokens")
plt.ylabel("Probabilities")
plt.xticks(rotation="vertical")
plt.show()

print("Logprobs: ", response["choices"][0]["logprobs"]["top_logprobs"][0])
