import openai
import os

# Set OpenAI API key
openai.api_key = ("APIKEY")

# Prompt for the AI to generate text from
prompt = "What vitamins should I take daily?"

# Generate text from the AI
response = openai.Completion.create(
    engine="davinci",
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

# Print the generated text
print(response.choices[0].text)

# In this code, we first set the OpenAI API key using an environment variable. Then we define a prompt that the AI will use to generate text from. We pass this prompt to the openai.Completion.create function, which sends the prompt to the OpenAI API and retrieves the generated text.

# The engine parameter specifies which OpenAI language model to use. In this case, we're using the davinci engine, which is the most powerful and capable OpenAI language model currently available.

# The max_tokens parameter sets the maximum number of tokens (words or symbols) that the AI is allowed to generate in response to the prompt. The n parameter specifies the number of text completions to generate. In this case, we're only generating one completion.

# The stop parameter specifies a string that the AI should stop generating text at. If we set stop to a punctuation mark like a period or a question mark, the AI will stop generating text at that point.

# The temperature parameter controls the "creativity" of the AI's responses. Higher temperatures will produce more diverse and unpredictable responses, while lower temperatures will produce more predictable and conservative responses. In this case, we're using a temperature of 0.5, which is a moderate value.

# Finally, we print the generated text by accessing the text property of the first element in the choices list of the response object.
