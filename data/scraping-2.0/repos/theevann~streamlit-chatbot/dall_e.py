import base64
import io

import streamlit as st
from openai import OpenAI
from PIL import Image


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
prompt_enforcer = "I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS:"

def st_dall_e():
    quality = st.sidebar.radio("Quality", ["standard", "hd"])
    size = st.sidebar.radio("Size", ["1024x1024", "1024x1792", "1792x1024"])
    style = st.sidebar.radio("Style", ["vivid", "natural"])
    enforce_prompt = st.sidebar.checkbox("Enforce prompt", value=False)

    st.title("DALL·E UI")
    prompt = st.text_area("Prompt for DALL·E", placeholder="A duck in a pond")

    if enforce_prompt:
        prompt = prompt_enforcer + " " + prompt

    if st.button("Generate", use_container_width=True, type="primary") and prompt != "":
        with st.spinner("Generating..."):
            response = client.images.generate(
                model="dall-e-3",
                response_format="b64_json",
                prompt=prompt,
                n=1,
                quality=quality,
                style=style,
                size=size
            )
            # image_url = response['data'][0]['url']
            # st.image(image_url)
            st.session_state.base64_image = response.data[0].b64_json
            st.session_state.revised_prompt = response.data[0].revised_prompt

    if "base64_image" in st.session_state:
        base64_image = st.session_state.base64_image
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        st.image(image)
        st.write("**Revised prompt**:", st.session_state.revised_prompt)

        # download button
        st.download_button(
            "Download image",
            io.BytesIO(image_data),
            "dall-e.png"
        )