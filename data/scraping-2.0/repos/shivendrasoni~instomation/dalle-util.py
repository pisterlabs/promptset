import openai

# Set the OpenAI API key
openai.api_key = "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"



def generate_image(prompt):
    # Set the phrase that you want to generate an image for
    prompt = "a cat playing with a dragon ball"

    # Use the Dall-E API to generate an image
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format="url"
    )

    # Print the URL of the generated image
    print(response.data[0])