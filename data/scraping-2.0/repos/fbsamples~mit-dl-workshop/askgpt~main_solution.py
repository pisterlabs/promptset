import openai
import yaml
import string
import questionary

# Set OpenAI API key
openai.api_key_path = 'apikey.file'

# Load prompt templates from a YAML file
with open('prompts.yaml') as f:
    PROMPTS = yaml.safe_load(f)

# Define a function to fill in the prompt template
def fill_prompt(task):
    prompt = PROMPTS[task]
    print(f"\nPrompt template: {prompt}\n")    
    # Parse the prompt template to find placeholders and ask the user to fill them in
    placeholders = {v[1]: "" for v in string.Formatter().parse(prompt) if v[1] is not None}
    for plac in placeholders.keys():
        placeholders[plac] = questionary.text(f"{plac}: ").ask()
    # Substitute the filled placeholders into the prompt template
    prompt = prompt.format(**placeholders)
    return prompt

# Define a function to generate a GPT response to a given prompt
def ask_gpt(prompt):
    print("\nGenerating response....\n")
    # Use OpenAI's Completion API to generate a response to the prompt
    response = openai.Completion.create(
        model="text-davinci-003",
        temperature=0.75,
        max_tokens=650,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        prompt=prompt
    )
    # Extract the GPT-generated answer from the API response
    gpt_answer = response['choices'][0]['text']
    return gpt_answer

# Main program entry point
if __name__ == "__main__":
    # Ask the user to select an action
    choices = PROMPTS.keys()
    print("\n\n")
    action = questionary.select("What do you want to do?", choices=choices).ask()
    if action == "freetext":
        # If the user selects "freetext", ask for a prompt
        prompt = questionary.text("Ask GPT: ").ask()
    else:
        # Otherwise, fill in the prompt template for the selected task
        prompt = fill_prompt(action)
    # Generate a GPT response to the prompt and print it
    response = ask_gpt(prompt)
    print(response)
