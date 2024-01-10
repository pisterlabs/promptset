import openai

openai.api_key = "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"
model_engine = "text-davinci-003"
response = openai.Image.create(
    engine=model_engine,
    prompt="a white siamese cat",   
    n=1,
    size="1024x1024"
)
image_url = response['data'][0]['url']
print(image_url)

"""response = openai.Image.create_edit(
  image=open("sunlit_lounge.png", "rb"),
  mask=open("mask.png", "rb"),
  prompt="A sunlit indoor lounge area with a pool containing a flamingo",
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']

response = openai.Image.create_variation(
  image=open("corgi_and_cat_paw.png", "rb"),
  n=1,
  size="1024x1024"
)
image_url = response['data'][0]['url']"""