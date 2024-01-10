# Text summarization
# One really common use case for using OpenAI's models is summarizing text. This has a ton of applications in business settings, including summarizing reports into concise one-pagers or a handful of bullet points, or extracting the next steps and timelines for different stakeholders.

# In this exercise, you'll summarize a passage of text on financial investment into two concise bullet points using a text completion model.

# The openai package has been pre-loaded for you.

# Instructions
# 100 XP
# Assign your API key to openai.api_key.
# Create a request to Completion endpoint, sending the prompt provided; use a maximum of 400 tokens and make the response more deterministic. 

# Set your API key
import openai
openai.api_key = "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"

prompt="""Summarize the following text into two concise bullet points:
Investment refers to the act of committing money or capital to an enterprise with the expectation of obtaining an added income or profit in return. There are a variety of investment options available, including stocks, bonds, mutual funds, real estate, precious metals, and currencies. Making an investment decision requires careful analysis, assessment of risk, and evaluation of potential rewards. Good investments have the ability to produce high returns over the long term while minimizing risk. Diversification of investment portfolios reduces risk exposure. Investment can be a valuable tool for building wealth, generating income, and achieving financial security. It is important to be diligent and informed when investing to avoid losses."""

# Create a request to the Completion endpoint
response = openai.Completion.create(
  model = "text-davinci-003",
  prompt = prompt,
  temperature = 0,
  max_tokens = 400
  

)

print(response["choices"][0]["text"])