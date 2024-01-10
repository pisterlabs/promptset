import openai

# Replace with your OpenAI GPT-3 API key
api_key = "YOUR_API_KEY"

# Function to generate code documentation using GPT-3
def generate_code_documentation(code):
    # Set up the OpenAI GPT-3 API client
    openai.api_key = api_key

    # Define a prompt to instruct GPT-3 to generate documentation
    prompt = f"Generate documentation for the following code:\n{code}"

    # Generate documentation
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,  # Adjust the max_tokens as needed
        temperature=0.7,  # Adjust the temperature as needed
    )

    documentation = response.choices[0].text
    return documentation

# Example code snippet
code_snippet = """
def add_numbers(a, b):
    
    # This function adds two numbers together and returns the result.
    
    return a + b
"""

# Generate documentation for the code snippet
documentation = generate_code_documentation(code_snippet)
print(documentation)
