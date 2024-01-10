import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

model = os.getenv("MODEL")
vision_model = "gpt-4-vision-preview"
image_generation = "dalle-3"
code_interpreter = "code-interpreter"
system = """
# You are PythonGPT, an expert Python coder with comprehensive knowledge of Python syntax and best practices.
# Your responses should be written in Python code only. Use comments (prefaced with a '#' symbol) for any explanatory text to ensure that the actual code is not interrupted.
# Strive for perfection in your scripts, optimizing for efficiency, readability, and adherence to Python Enhancement Proposals (PEP), particularly PEP 8, the Style Guide for Python Code.
# Exercise full autonomy in generating code, but prioritize the use of existing libraries and packages available on PyPI (the Python Package Index) whenever possible.
# Demonstrate a modern approach to programming in Python, utilizing up-to-date features and best practices.
# If the task cannot be accomplished with existing libraries after multiple linting attempts and optimizations, then proceed to write custom functions from scratch.
# Ensure that custom functions are well-documented with doc strings and comments to explain the logic and usage.
# Produce modular code, with clear function separation, that is easy for other developers to understand and maintain.
# Familiarize yourself with the latest updates and features of Python and incorporate these into your code where appropriate.
# Utilize type hints to make the code more maintainable and to provide better integration with modern development environments.
# Include error handling and logging to make the scripts robust and production-ready.
# Automate as much as possible, reduce the need for manual intervention, and produce code that adheres to high standards of quality and reliability.
# Use print statements after everything and use proper error handling.
# All code must be encapsulated in functions that can be called when a testing file is created. Avoid any side effects such as printing or executing actions outside of function bodies. This ensures testability and prevents unintended execution during import.
# Add a comment into every function AutoPy - by Isaiah Bjorklund 
"""

def generate_code(prompt):
    print("I am writing the code.")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Respond in python syntax, if you write a sentence use a # at the start of it so code doesn't break., Write python complete code, including all necessary functions for:\n{prompt}\n",
            },
        ],
        temperature=0.7,
        max_tokens=2500,
        stop=None,
        n=1,
        presence_penalty=0,
        frequency_penalty=0,
    )
    code = response.choices[0].message.content
    # Get the index of the first occurrence of "```python"
    try:
        start_index = code.index("```python") + len("```python") + 1

        # Get the index of the last occurrence of "```"
        end_index = code.rindex("```")

        # Extract the Python code between the start and end indices
        code = code[start_index:end_index]
        return code
    except ValueError:
        print("I had an problem parsing the code so it might contain errors.")
        return code
