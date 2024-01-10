import os
import openai


def add_full_stops_to_comments(input_code):
    """Take some code and add full stops to comments."""
    # Get the OpenAI API key.
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Format the prompt.
    prompt = input_code + "\n\n% Refactor this code to add full stops to the end of each comment block if a full " \
                          "stop is not present."

    # Make a call to the OpenAI API to modify the text.
    response = openai.Completion.create(
        engine='text-davinci-003',  # Choose the desired language model
        prompt=prompt,  # Provide the code as the prompt
        max_tokens=2000,  # Set the maximum number of tokens in the response
        temperature=0.0,  # Control the randomness of the response
        n=1,  # Number of completions to generate
        stop=None,  # Stop generating completions at specified tokens (optional)
    )

    # Extract the text from OpenAI's response.
    modified_code = response.choices[0].text.strip()  # Extract the modified code from the API response
    return modified_code


if __name__ == '__main__':
    pass
