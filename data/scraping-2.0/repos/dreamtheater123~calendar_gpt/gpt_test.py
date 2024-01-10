# Import the os package
import os

# Import the openai package
import openai

# From the IPython.display package, import display and Markdown
from IPython.display import display, Markdown

# Import yfinance as yf
import yfinance as yf

# Set openai.api_key to the OPENAI environment variable
openai.api_key = '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'

response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[{"role": "system", "content": 'assist the request by the user'},
                        {"role": "user", "content": 'What is passport?'}
              ])
print(response)
