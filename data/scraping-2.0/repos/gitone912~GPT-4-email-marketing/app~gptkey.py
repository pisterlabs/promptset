import openai


def generate_prompt(prompt):
# Set up your OpenAI API key
    openai.api_key = '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'

    
    # Generate response
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
    )

    # Extract the generated text from the response
    generated_text = response.choices[0].text.strip()

    # Print the generated text
    
    return generated_text