import openai

# Set your OpenAI API key
api_key = "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"
openai.api_key = api_key

# Provide the code that you want to document
code_to_document = """
def calculate_square(n):
    # This function calculates the square of a given number.
    return n * n
"""

# Generate documentation using GPT-3
def generate_code_documentation(code):
    prompt = f"Generate documentation for the following code:\n\n{code}\n\nDocumentation:"

    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150  # Adjust the token limit as needed
    )

    return response.choices[0].text

# Generate documentation for the code
documentation = generate_code_documentation(code_to_document)
print(documentation)
