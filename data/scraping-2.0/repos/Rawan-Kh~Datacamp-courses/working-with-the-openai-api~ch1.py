# Import openai
import openai

# Set your API key
openai.api_key = ""

# Create a request to the Completion endpoint
response = openai.Completion.create(
  # Specify the correct model
  model="gpt-3.5-turbo-instruct",
  prompt="Who developed ChatGPT?"
)

print(response)

------------
# {
#   "id": "cmpl-7CTIdXA6C2rENhDBNHxHkynyYRWZS",
#   "object": "text_completion",
#   "created": 1683206919,
#   "model": "text-davinci-003",
#   "choices": [
#     {
#       "text": "\n\nThe goal of OpenAI is to advance digital intelligence in the way that is most likely to benefit humanity as a whole. OpenAI works to build safe artificial general intelligence (AGI) and ensure that AGI's benefits are as widely and equitably distributed as possible. OpenAI also works to research and develop friendly AI, which focuses on how to develop AI systems that align with human values and goals.",
#       "index": 0,
#       "logprobs": null,
#       "finish_reason": "stop"
#     }
#   ],
#   "usage": {
#     "prompt_tokens": 8,
#     "completion_tokens": 83,
#     "total_tokens": 91
#   }
# }
# Extract the model from the response
print(response["model"])
# Extract the total_tokens from the response
print(response['usage']['total_tokens'])
# Extract the text from the response
print(response['choices'][0]['text'])
--------

# What are the benefits of having separate organizations for each business unit or product feature?
# Removes single failure point
# Improved management of usage and billing
