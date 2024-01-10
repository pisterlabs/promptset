from transformers.tools import OpenAiAgent
from transformers import Tool
from e2b import CodeInterpreter
import os
import json

e2b_key = os.environ.get('E2B_API_KEY')
openai_key = os.environ.get('OPENAI_API_KEY')

class E2BCodeInterpreter(Tool):
    name = "e2b_code_interpreter"
    description = ("This tool creates an isolated environment to run python code. It takes the python code as input and returns the output of the code.")
    inputs = ['text']
    outputs = ['text']

    def __call__(self, code: str):
        sandbox = CodeInterpreter()
        stdout, stderr, artifacts = sandbox.run_python(code)
        sandbox.close()
        if stderr:
            return "There was following error during execution: " + stderr
        
        return json.dumps(stdout)

e2b_code_interpreter = E2BCodeInterpreter()

agent = OpenAiAgent(
    model='gpt-3.5-turbo', api_key=openai_key, additional_tools=[e2b_code_interpreter]
)

code = """
import matplotlib.pyplot as plt

months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
sales = [120, 135, 150, 160, 155, 165, 170, 165, 160, 170, 180, 190]
max_sales = max(sales)
print("Max sales:", max_sales)

plt.figure(figsize=(10, 6))
plt.plot(months, sales, marker='o', color='b', linestyle='-')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Sales (in thousands)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
"""

agent.run(f"Run the following code using e2b code interpreter: { code }.")