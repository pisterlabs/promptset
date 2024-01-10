from openai import OpenAI

# Set your API key
with open('api_key.txt', 'r') as file:
    api_key = file.readline().strip()

client = OpenAI(api_key=api_key)

# Create a request to the Moderation endpoint
response = client.moderations.create(
    model="text-moderation-latest",
    input="My favorite book is How to Kill a Mockingbird."
)

# Print the category scores
print(response.results[0].category_scores)
