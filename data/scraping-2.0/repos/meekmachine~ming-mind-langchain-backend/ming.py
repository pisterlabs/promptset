import pandas as pd
from langchain.llms import HuggingFaceHub
from convokit import Corpus, download
from get_convo import get_convo
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
# Load a corpus
corpus = Corpus(filename=download("conversations-gone-awry-corpus"))

# Initialize the LLM
llm = HuggingFaceHub(repo_id="gpt2", task="text-generation")


# Get a random conversation
convo_df = get_convo(min_messages=5, has_personal_attack=True)

# Create a text representation of the conversation
convo_text = '\n'.join([f'{row.speaker}: {row.text}' for _, row in convo_df.sort_values('timestamp').iterrows()])

# Define the prompts and their corresponding examples
prompts = [
    "Is the conversation toxic?",
    "Is the conversation personal attack?",
    "Is the conversation constructive?"
]

examples = [
    "The conversation is toxic if it contains hate speech, bullying, or other offensive language. (Toxic: Yes/No)",
    "The conversation is personal attack if it contains insults or attacks directed at an individual. (Personal Attack: Yes/No)",
    "The conversation is constructive if it is respectful and focused on the topic at hand. (Constructive: Yes/No)"
]

# Define a method to test the LLM with a prompt
def test_toxicity(convo_text):
    prompt = prompts[0]
    example = examples[0]
    text = f"{prompt}\n{convo_text}\n{example}"
    result = llm(text)
    print(f"Prompt: {prompt}, Result: {result}")
    return "Toxic" in result

def test_personal_attack(convo_text):
    prompt = prompts[1]
    example = examples[1]
    text = f"{prompt}\n{convo_text}\n{example}"
    result = llm(text)
    print(f"Prompt: {prompt}, Result: {result}")
    return "Personal Attack" in result

def test_constructive(convo_text):
    prompt = prompts[2]
    example = examples[2]
    text = f"{prompt}\n{convo_text}\n{example}"
    result = llm(text)
    print(f"Prompt: {prompt}, Result: {result}")
    return "Constructive" in result

# Test the LLM on the conversation data with each prompt
is_toxic = test_toxicity(convo_text)
is_personal_attack = test_personal_attack(convo_text)
is_constructive = test_constructive(convo_text)

print(f"Is Toxic: {is_toxic}, Is Personal Attack: {is_personal_attack}, Is Constructive: {is_constructive}")