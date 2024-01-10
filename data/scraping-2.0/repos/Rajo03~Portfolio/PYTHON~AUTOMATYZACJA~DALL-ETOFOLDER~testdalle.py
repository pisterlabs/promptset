import openai

# Your DALL-E API key
openai.api_key = "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"

# The text prompt you want to use to generate an image
prompt = "A moose in the snow"

# Generate an image
response = openai.Image.create(
    prompt=prompt,
    model="image-alpha-001",
    size="1024x1024",
    response_format="url"
)

# Print the URL of the generated image
print(response["data"][0]["url"])