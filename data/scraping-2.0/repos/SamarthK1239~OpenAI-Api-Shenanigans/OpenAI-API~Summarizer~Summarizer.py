import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import TokenSplitter

path = Path("../Environment-Variables/.env")
load_dotenv(dotenv_path=path)

# Set up openai client
openai = OpenAI(
    organization=os.getenv('organization'),
    api_key=os.getenv("api_key")
)

# Read transcription file
with open("transcription.txt") as f:
    transcription = f.readline()

# Parameter Meanings for response generation
# temperature: Controls Randomness. Lower means less random completions. As this value approaches zero, the model becomes very deterministic
# max_tokens: Maximum of 4000 tokens shared between prompt and completion (input and output)
# top_p: Controls diversity. 0.5 means half of all weighted options are considered
# frequency_penalty: Penalizes new tokens based on frequencies. Decreases the chances of repetition of the same lines
# presence_penalty: Penalizes new tokens based on if they show up already. Increases the likelihood of new topics coming up
# best_of: Generates the specified number of items and then returns the best one

prompt = "Comprehensively summarize this for a university student. Using bullet points to organize the summary, " \
         "Go through every piece of advice provided by the speaker. " \
         "If you can use technical programming terms, be sure to reference them.\n" + transcription

# First generation pass using davinci-003 model
response = openai.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "user", "content": prompt},
    ]
)

print(response.choices[0].message.content)

# Fact Checking pass, uses same model as above
fact_checked_response = openai.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "user", "content": "Clarify each bullet point: "},
        {"role": "user", "content": response.choices[0].message.content}
    ]
)
print(fact_checked_response.choices[0].message.content)

# Detail-addition pass, using same model as above
final_detailed_response = openai.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    messages=[
        {"role": "user", "content": "Add as much detail as you can to each bullet point. Use paragraphs to organize your response."},
        {"role": "user", "content": fact_checked_response.choices[0].message.content}
    ]
)
print(final_detailed_response.choices[0].message.content)

# Print final response after all three passes
print("Final Result:", final_detailed_response.choices[0].message.content)
