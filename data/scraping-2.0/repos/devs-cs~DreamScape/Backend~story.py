import openai
import json

# Load OpenAI API credentials from a file
with open('openai_api.json', 'r') as file:
    api_keys = json.load(file)
    openai.api_key = api_keys.get("OPENAI_API_KEY")

def generate_story(input_strings):
    # Prepare an advanced prompt for the OpenAI model based on the input strings
    elements = ", ".join(input_strings)
    prompt = (
        f"Create a simple story suitable for text-to-video conversion, including the following elements: {elements}. "
        "The story should have a clear beginning, middle, and end. "
        "Begin with an introduction of the setting and characters, followed by a problem, and then resolve the problem in the end. "
        "Make it short and 3 lines long. Print each of the three lines in a new line. "
    )

    # Use OpenAI to generate a continuation of the prompt
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use "gpt-4" if available, otherwise use "gpt-3.5-turbo"
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )

    story = response['choices'][0]['message']['content'].strip()
    return story


if __name__ == "__main__":
    main()
