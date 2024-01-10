import subprocess
import openai

# Set your OpenAI GPT-3 API key
api_key = "Code_Narrator"
openai.api_key = "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"

# Define the Python code you want to analyze
python_code = """
def add_numbers(a, b):
    return a + b
"""

# Step 1: Code Analysis with Pylint
def run_pylint(code):
    try:
        # Run Pylint on the code and capture the output
        output = subprocess.check_output(["pylint", "-"], input=code.encode(), text=True, stderr=subprocess.STDOUT)
        return output
    except subprocess.CalledProcessError as e:
        return e.output

# Step 2: Document the Code using GPT-3
def document_code_with_gpt3(code):
    try:
        # Define a query for GPT-3 to generate documentation
        query = f"Document the following Python code:\n\n{code}"
        
        # Use GPT-3 to generate documentation
        response = openai.Completion.create(
            engine="davinci",
            prompt=query,
            max_tokens=100
        )
        return response.choices[0].text
    except Exception as e:
        return str(e)

# Step 1: Analyze the code with Pylint
python_code='''def generate_fibonacci(n):
    fibonacci_series = []
    a, b = 0, 1
    while len(fibonacci_series) < n:
        fibonacci_series.append(a)
        a, b = b, a + b
    return fibonacci_series

# Change the value of 'n' to the number of Fibonacci terms you want
n = 10  # You can replace this with your desired number of terms
fib_series = generate_fibonacci(n)

# Print the Fibonacci series
print("Fibonacci Series:")
for number in fib_series:
    print(number)'''

pylint_output = run_pylint(python_code)

# Step 2: Generate documentation using GPT-3
documentation = document_code_with_gpt3(python_code)

print("Pylint Output:")
print(pylint_output)

print("\nGenerated Documentation:")
print(documentation)
