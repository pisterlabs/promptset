import json 
import openai
import os

openai.api_key = os.getenv("CHATGPT_KEY")

architecture_required = "dog walking"
architecture_required_path = architecture_required.replace(" ","_")
version = "1"

template = """
---
name: Shopping
summary: |
  Domain for everything shopping
owners:
    - team1
    - team2
---

<Admonition>Domain for everything to do with Shopping at our business. Before adding any events or services to this domain make sure you contact the domain owners and verify it's the correct place.</Admonition>

### Details

This domain encapsulates everything in our business that has to do with shopping and users. This might be new items added to our online shop or online cart management.

<NodeGraph title="Domain Graph" />
"""

file=open(f"event_lists/{architecture_required_path}/{architecture_required_path}_v{version}.json","r")

text = file.read()

event_list = json.loads(text)

unique_domains = set(d['domain'] for d in event_list)
unique_teams = set(d['team'] for d in event_list)

for domain in unique_domains:
    print(domain)
    chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": f"Return the markdown only in your response for the domain {domain} in the context of {architecture_required} using this markdown as the example template {template} with owner values being the appropriate values only from this list {unique_teams}"}])
    
    out = chat_completion['choices'][0]['message']['content']

    domain_path = domain.replace(" ","_")

    with open(f"domains/{architecture_required_path}_v{version}/{domain_path}/index.md", "w") as outfile:
        outfile.write(out)

