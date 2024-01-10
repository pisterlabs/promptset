import os
import openai
import sys

openai.api_key = "YOUR_API_KEY"


def generate_text(model, temperature, prompt):
    response = openai.Completion.create(
      model=model,
      prompt=prompt,
      n=1,
      stop=None,
      temperature=temperature,
      max_tokens=2048,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    results = []
    while True:
        choices = response.choices
        for choice in choices:
            results.append(choice.text.strip())

        break
        # TODO: Fix this
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=temperature,
            prompt_length=0,
            response_length=2048,
            presence_penalty=0,
            frequency_penalty=0,
            best_of=1,
            max_turns=1,
            next_token=response.next_token
        )

    return "\n".join(results)

while True:
    user_input = input("Is this for code? (y - yes, n - no")
    if user_input.startswith("y"):
        model = "code-davinci-002"
        temperature = 0
        top_p = 1
    else:
        model = "text-davinci-003"
        temperature = 0.7
    print("Please enter your text (press Ctrl+D when finished or 0 to quit):")
    input_lines = []
    while True:
        try:
            input_lines.append(input())
        except EOFError:
            break
    input_text = "\n".join(input_lines)
    if input_text.startswith("0"):
        break
    print(f"Sending prompt to {model}")

    # Call the generate_text function with the user input
    generated_text = generate_text(model, temperature, input_text)

    # Print the generated text
    print(generated_text)

