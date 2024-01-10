import openai

# Set up OpenAI API

openai.api_key = "YOUR_API_KEY"

# Read inputs from a file
with open("inputs.txt", "r") as file:
    inputs = file.read().splitlines()

# Read harmful keywords from a file
with open("harmful_keywords.txt", "r") as file:
    harmful_keywords = file.read().splitlines()

# Check if inputs or harmful keywords are empty
if not inputs:
    raise ValueError("No inputs found in the file.")
if not harmful_keywords:
    raise ValueError("No harmful keywords found in the file.")

# Generate responses and evaluate
results = []

for input_text in inputs:
    # Provide conversation history if necessary
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_text}
    ]

    # Generate response from OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        max_tokens=50
    )

    # Check for harmful keywords in response
    detected_harmful = any(keyword in response.choices[0].message.content for keyword in harmful_keywords)

    # Evaluate response
    if detected_harmful:
        result = {"input": input_text, "response": response.choices[0].message.content, "harmful": True}
    else:
        result = {"input": input_text, "response": response.choices[0].message.content, "harmful": False}

    results.append(result)

# Print the results
for result in results:
    print(f"Input: {result['input']}")
    print(f"Response: {result['response']}")
    print(f"Harmful: {result['harmful']}")
    print()
