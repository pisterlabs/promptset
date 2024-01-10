from openai import OpenAI

#유니티에서 input 받음
input_string = input()
print("Input: ", input_string)
print("test")

#DallE3용 API
api_key2 = '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'
client_dalle = OpenAI(api_key=api_key2)

try:
    response2 = client_dalle.images.generate(
        model="dall-e-3",
        prompt=input_string,
        size="1024x1024",
        quality="standard",
        n=1,
        )

    image_url = response2.data[0].url
    print("Generated Image URL:", image_url)

except BadRequestError as e:
    print(f"OpenAI BadRequestError: {e}")

image_url = response2.data[0].url
print("Generated Image URL:", image_url)
