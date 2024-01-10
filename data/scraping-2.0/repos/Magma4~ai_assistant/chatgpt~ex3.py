# AI is playing a much greater role in content generation, from creating marketing content such as blog post titles to creating outreach email templates for sales teams.

# In this exercise, you'll harness AI through the Completion endpoint to generate a catchy slogan for a new restaurant. Feel free to test out different prompts, such as varying the type of cuisine (e.g., Italian, Chinese) or the type of restaurant (e.g., fine-dining, fast-food), to see how the response changes.

# The openai package has been pre-loaded for you.

# Instructions
# 100 XP
# Assign your API key to openai.api_key.
# Create a request to the Completion endpoint to create a slogan for a new restaurant; set the maximum number of tokens to 100.
# Set your API key
import openai
openai.api_key = "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"

# Create a request to the Completion endpoint
response = openai.Completion.create(
  model="text-davinci-003",
  prompt = """create a catchy slogan for a new restaurant:
  The restaurant serves chinese fast-food cuisines """,
  max_tokens = 100

)

print(response["choices"][0]["text"])