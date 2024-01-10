from openai import OpenAI

import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize the OpenAI client
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {
      "role": "user",
      "content": "I'm the VP of Finance Transformation at a major telecommunications provider. Help me define the use case for \"Supplier Performance and Supplier Risk Management\" where AI can drive process efficiency and insights. Describe to me the key problem, stakeholders, specific business challenges, business value and KPIs imacted, and AI capabilities needed to drive the transformation"
    }
  ],
  temperature=0.9,
)

print(response.choices[0])