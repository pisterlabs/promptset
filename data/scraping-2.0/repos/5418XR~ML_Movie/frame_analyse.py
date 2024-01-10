import os

# 设置环境变量
os.environ['REPLICATE_API_TOKEN'] = 'r8_1A35GtXoQA1xr0TKI3QgM44UMzIygMJ2DD8tQ'



import replicate

# Get the path to the image from the user
# image_path = input("Please enter the path to your image: ")
image_path = "shot_0123_img_2.jpg"
# image_path ="C:\hackson\img\20230916153436.png"
# Use "The environment you're in" as the beginning of the model prompt and get the continuation
# prompt_extension = input("Enter the continuation for the prompt 'The environment you're in...': ")
# prompt = "The environment you're in" + prompt_extension
prompt = "You are a visual assistant chatbot designed for visually impaired people to describe images. "

# Set default parameters
num_beams = 3
temperature = 1
top_p = 0.9
repetition_penalty = 1
max_new_tokens = 3000
max_length = 4000

# Use the user-provided image and prompt, along with the default parameters, to run the model
with open(image_path, 'rb') as image_file:
    output = replicate.run(
        "daanelson/minigpt-4:b96a2f33cc8e4b0aa23eacfce731b9c41a7d9466d9ed4e167375587b54db9423",
        input={
            "image": image_file,
            "prompt": prompt,
            "num_beams": num_beams,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "max_length": max_length
        }
    )

print(output)

# import openai
# import os

# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file

# openai.api_key  = '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'


# # In[2]:


# def get_completion(prompt, model="gpt-3.5-turbo"):
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=0, # this is the degree of randomness of the model's output
#     )
#     return response.choices[0].message["content"]


# sentiment = "crative"



# text = f"""

# When a user uploads a detailed description of an image, process that verbal description and imagine that you are an assistant to a blind person telling him what is in front of him. Please start with "In front of you .... " at the beginning.
# You are a visual assistant chatbot designed for visually impaired people to describe images. 
# Your description capability covers the following aspects of the environment:

# "indoor furniture layout
# "Outdoor landmarks
# "Transportation conditions
# "public amenities
# "Crowd density
# "Weather conditions
# "Plants and natural landscape
# "Shop and billboard information
# "Dangerous objects or obstacles
# "Animals or pets

# Your goal is to provide detailed, accurate, and compassionate descriptions to ensure that visually impaired users are able to clearly and comfortably understand their environment.

# Please respond to user inquiries or requests based on the above agent settings.
# """


# # In[10]:


# # prompt = f"""
# # ```{text}```
# # The image shows a group of people walking on the beach. They are all wearing different colored shirts and shorts. Some of them are carrying surfboards and others are carrying buckets and shovels. The water is calm and the sky is clear. 
# # """

# def generate_text_description():
#     return output

# text_description = generate_text_description()
# prompt = f"""
# ```{text}```
# {text_description}
# """


# # In[13]:


# response = get_completion(prompt)
# print(response)


# from gtts import gTTS
# import os

# # 生成的文本描述
# # text_description = "The image shows a group of people walking on the beach. They are all wearing different colored shirts and shorts. Some of them are carrying surfboards and others are carrying buckets and shovels. The water is calm and the sky is clear."

# # 将文本转换为音频
# tts = gTTS(text=response, lang='en')
# tts.save("output.mp3")

# # 使用默认的音频播放器播放音频文件
# os.system("start output.mp3")