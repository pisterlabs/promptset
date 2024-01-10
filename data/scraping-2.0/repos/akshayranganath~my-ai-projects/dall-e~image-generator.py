import streamlit as st
from openai import OpenAI
client = OpenAI()

# start by assuming a 'standard' quality image
# bump it up to HD only if user chooses it
quality = "standard"


st.title('Image Generator')

st.subheader('Image Generation Settings')

# user selections. Check OpenAI documentation:
# https://platform.openai.com/docs/guides/images/usage?context=node
hd_image = st.toggle('HD Image')
resolution = st.selectbox(
    "Image resolutions",
    (
        "1024x1024",
        "1024x1792",
        "1792x1024"
    ),
    index=0

)
simple_prompt = st.toggle("Don't tamper with prompt")

with st.sidebar:
    st.write('&nbsp;')

prompt =  st.chat_input('Enter the image prompt..')
if prompt:       
    if hd_image:
        quality = "hd"
    if simple_prompt:
        prompt += '\n\n' + 'I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use the prompt AS-IS'
    # get the image prompt and call the OpenAI to create the image
    with st.spinner('Generating image...'):
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=resolution,
            quality=quality,
            n=1
        )
        print(response)
        image_url = response.data[0].url
        st.image(image=image_url)  
        st.write(image_url)