import os

import openai
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

os.environ["OPENAI_API_KEY"] = "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"

# question = "what are the products offer by EngageBay"
# question = "testimonials of EngageBay"
# question = "how many badges EngageBay have"
# question = "list all EngageBay badges"
# question = "list all EngageBay badges links"
# question = "what services do you offer"
# question = "tell me EngageBay HAPPY CUSTOMERS REVIEWS"
question = "what Support does EngageBay offer"

loader = TextLoader('langchain-input.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

doc = index.query_with_sources(question)["answer"]

prompt_text = f'{doc}:{question}:'
message_log = [{"role": "user", "content": prompt_text}]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=message_log,
    max_tokens=150,
    stop=None,
    temperature=0.7,
)

print(response.choices[0].message.content)