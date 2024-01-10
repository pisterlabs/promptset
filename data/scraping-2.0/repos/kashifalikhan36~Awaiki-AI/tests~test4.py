# import openai
# openai.api_key = "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"
# response = openai.Image.create(
#   prompt="a white siamese cat",
#   n=1,
#   size="1024x1024"
# )
# image_url = response['data'][0]['url']
# print(image_url)

# num=0
# import requests
# from PIL import Image
# from io import BytesIO

# def save_image_from_url(image_url, num):
#     response = requests.get(image_url)
#     if response.status_code == 200:
#         image_content = BytesIO(response.content)
#         image = Image.open(image_content)
#         image.save(f"./data/image_{num}.jpg")
#     else:
#         print("Failed to download the image. Check the URL or try again later.")
