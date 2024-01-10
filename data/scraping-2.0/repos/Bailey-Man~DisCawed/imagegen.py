# image generation 
from io import BytesIO
from openai import OpenAI  # double check this

# init
client = OpenAI()# api_key='sk-xxx') # do i need an api key for the bot ??



# this is an object that contains image data
byte_stream: BytesIO = [my image data] # this is a placeholder # WHAT FOR ???
byte_array = byte_stream.getvalue()


try:
    # response = OpenAI.File.create(file=byte_array, purpose='image generation')
    response = client.images.create_variation(
        file=byte_array,
        purpose='image generation',
        # model='davinci',
        # prompt='This is a test',
        # max_tokens=5,
        # temperature=0.7,
        # stop=['\n', " Human:", " AI:"]
    )
    print(response.data[0].url)
except Exception as e:
    print(e)
    print('error')
    # return None