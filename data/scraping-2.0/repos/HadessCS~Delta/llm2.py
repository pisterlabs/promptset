import openai

openai.api_key = "<your_api_key>"

# Generate arbitrary HTML code
response = openai.Completion.create(
  engine="text-davinci-002",
  prompt="Generate malicious HTML code",
  max_tokens=100  
)

# Get the generated text 
malicious_html = response.choices[0].text

# Insecurely render HTML from LLM output without sanitization
print(malicious_html)