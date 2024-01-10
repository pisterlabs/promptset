# Import OpenAI library
import openai

# Set your API key
openai.api_key = "<your API key>"

# Choose a ChatGPT model
model = "chatgpt:small"

# Provide your text input
text = "Hello, how are you?"

# Make a request to ChatGPT API endpoint
response = openai.Chat.create(
    model=model,
    text=text,
    chat_id="my_chat"
)

# Print the generated text output
print(response.output)

# to do: how to get api_key
# to do: test other model
# to do: intergration with tianmaojingling