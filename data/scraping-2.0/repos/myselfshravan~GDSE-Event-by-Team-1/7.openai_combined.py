import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

st.set_page_config(page_title="OpenAI API", page_icon="🤖")


def generate_text(prompt, model_id):
    completion = openai.Completion.create(
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        model=model_id
    )
    response = completion.choices[0].text
    return response


def generate_image(prompt):
    completion = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    image_url = completion.data[0].url
    return image_url


st.title("Gen AI 🤖")
st.subheader("Write Prompt below")
prompt = st.text_area("Prompt", "")
engine = st.radio("Select Engine", ("text", "image"))

model_selected = None
if engine == "text":
    complete_list = openai.Model.list()
    model_list = [model["id"] for model in complete_list["data"]]
    model_selected = st.selectbox("Select Model", model_list)

if st.button("Generate"):
    if engine == "text":
        response = generate_text(prompt, model_selected)
        st.write(response)
    elif engine == "image":
        image_url = generate_image(prompt)
        st.image(image_url, use_column_width=True)
