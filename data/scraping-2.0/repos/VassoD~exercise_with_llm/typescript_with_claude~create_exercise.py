import os
from anthropic import Anthropic
import random

api_key = os.getenv("ANTHROPIC_API_KEY")

anthropic = Anthropic(api_key=api_key)

prompt = """
Human: Please think of a TypeScript coding challenge to test an advanced level developer his understanding of a concept.

The concepts can include:
<concepts>
  - types
  - arrays
  - any and unknown
  - another typescript concept
</concepts>

Please choose one of the above concepts randomly.

The coding challenge should:
- Tell which concept is being tested
- Be complex enough to test the developer's understanding of the concept
- Provide a short code snippet to initalize the problem, but not the solution

Assistant: 
"""

# Use Anthropic to generate the Typescript exercise code based on the prompt
response = anthropic.completions.create(
    model="claude-2",
    prompt=prompt,
    max_tokens_to_sample=150
)

# Extract the generated code from the response
exercise_code = response.completion

# Save the exercise code to a Typescript file
exercise_file_path = "exercise2.ts" 
if os.path.exists(exercise_file_path):
  counter = 1
  while os.path.exists(f"exercise{counter}.ts"):
    counter += 1
  exercise_file_path = f"exercise{counter}.ts"

with open(exercise_file_path, "w") as exercise_file:
    exercise_file.write(exercise_code)

print(f"Exercise file '{exercise_file_path}' generated.")
