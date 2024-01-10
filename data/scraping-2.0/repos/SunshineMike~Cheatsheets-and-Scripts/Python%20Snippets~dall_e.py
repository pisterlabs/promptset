import openai

class DallE:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def generate(self, prompt, n, size):
        if size == "1":
            dimension = "256x256"
        elif size == "2":
            dimension = "512x512"
        elif size == "3":
            dimension = "1024x1024"
        else:
            print("Invalid size")
            return

        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=dimension
        )
        image_url = response['data'][0]['url']
        return image_url


'''
generator = ImageGenerator("your_openai_api_key_here")
image_url = generator.generate("A beautiful sunset over the ocean", 1, 2)
print(image_url)
'''