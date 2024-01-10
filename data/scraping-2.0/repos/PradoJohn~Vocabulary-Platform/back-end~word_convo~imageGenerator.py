import openai

openai.api_key = ""

def generate_image(prompt):
    try:
        response = openai.Image.create(
            n=1,
            prompt=prompt,
        )
        print(response)  # Print the entire response for debugging
        print(response.assets[0].url)
        return response.assets[0].url
    except Exception as e:
        print(f"Error in generate_image: {e}")
        return None 


if __name__ =='__main__':
    generate_image()
  # generate_image(
  #    'Generate an image with dimensions 800x1200 based on the prompt:  A man standing in front of a luxurious car with a satisfied expression, holding a golden key that symbolizes the exclusive privilege of owning a premium vehicle.')