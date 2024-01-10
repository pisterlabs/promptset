from anthropic import Anthropic
import os
import sys

# Get your API key from an environment variable
api_key = os.getenv("ANTHROPIC_API_KEY")

anthropic = Anthropic(api_key=api_key)

if len(sys.argv) > 3:
    print("Usage: python script.py <response_file_path>")
    sys.exit(1)

response_file_path = sys.argv[1]
with open(response_file_path, "r") as response_file:
    response_code = response_file.read()

prompts = {
    "claude": {
        "human": "Claude, please provide a Typescript exercise for me:",
        "ai": "Response format for exercise:\n\n{exercise}\n\nExplanation:\n{explanation}\n\nBest Practices:\n- Explain best practices for the solution."
    }
    # Define other prompts as needed for different roles or contexts
}

role = "claude"

# Analyze the code with prompts
def analyze_code(code, role):
    if role == "claude":
        HUMAN_PROMPT = "You are Claude, the Typescript instructor. Your role is to provide a Typescript exercise and analyze the response."
        AI_PROMPT = "Response format for exercise:\n\n{exercise}\n\nExplanation:\n{explanation}\n\nBest Practices:\n- Explain best practices for the solution."
    else:
        HUMAN_PROMPT = "Please review this code snippet in the context of the project:"
        AI_PROMPT = prompts[role]['ai']

    prompt = f"""
\n\nHuman: {HUMAN_PROMPT} {prompts[role]['human']}\n\n{code}\n\nIn your response:\n- List any errors or issues found in the code\n- Provide 2-3 suggestions for improvement\n- Share links to relevant coding standards/documentation\n- Consider the context.\n\nUse simple language a beginner would understand\n\nAI: {AI_PROMPT}
\n\nAssistant:
"""

    response = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=500,
        prompt=prompt
    ).completion

    return response

analysis = analyze_code(response_code, role)

# Save the analysis to an MD file
with open("analysis.md", "w") as analysis_file:
    analysis_file.write("## Code Analysis\n\n")
    analysis_file.write(f"Analysis for the Typescript exercise response:\n\n")
    analysis_file.write(analysis)

print("Code analysis saved to 'analysis.md'.")

