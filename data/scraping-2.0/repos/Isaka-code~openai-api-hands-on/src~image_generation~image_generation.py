from openai import OpenAI

def generate_image(model, prompt):
    client = OpenAI()

    response = client.images.generate(
        model=model,
        prompt=prompt,
        n=1,
        size="1024x1024"
        )

    return response

if __name__ == "__main__":
    model="dall-e-3"
    prompt="A Pokemon resembling a cute baby sea otter"

    response = generate_image(model, prompt)

    print(response)
