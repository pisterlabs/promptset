from openai_module import client

async def generate_dall_e_image(prompt: str, n: int=1, size: str = "256x256", model: str = "dall-e-2"):
    response = client.images.generate(
        model=model,
        prompt=prompt,
        n=n,
        response_format="url",
        size=size,
        quality="standard",
        style="vivid"
    )
    url = response.data[0].url
    print(f"dall-e url in generate_dall_e_image: {url}")
    return url