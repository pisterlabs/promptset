import streamlit as st
from langchain.llms import Ollama

from multimodal_ollama import convert_to_base64, plt_img_base64

# wide mode
st.set_page_config(layout="centered")

st.title("Chat with Images locally hosted")
st.subheader("Multi-modal LLMs. Streamlit + Ollama + LangChain = 🚀🫶")

# choose model
model = st.selectbox("Choose a model", ["llava", "llava:13b", "bakllava", "bakllava:7b"])
st.session_state["model"] = model

# chatbot stuff
st.markdown("---")

# upload images
def upload_image():
    images = st.file_uploader("Upload an image to chat about", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    # assert max number of images, e.g. 7
    assert len(images) <= 7, (st.error("Please upload at most 7 images"), st.stop())

    if images:
        # convert images to base64
        images_b64 = []
        for image in images:
            image_b64 = convert_to_base64(image)
            images_b64.append(image_b64)

        # display images in multiple columns
        cols = st.columns(len(images_b64))
        for i, col in enumerate(cols):
            col.markdown(f"**Image {abs((i+1)-len(cols))+1}**")
            col.markdown(plt_img_base64(images_b64[i]), unsafe_allow_html=True)
        st.markdown("---")
        return images_b64
    st.stop()


# init session state of the uploaded image
image_b64 = upload_image()


# ask question
q = st.chat_input("Ask a question about the image(s)")
if q:
    question = q
else:
    # if isinstance(image_b64, list):
    if len(image_b64) > 1:
        question = f"Describe the {len(image_b64)} images:"
    else:
        question = "Describe the image:"


# run model
@st.cache_data(show_spinner=False)
def run_llm(question, image_b64, model):
    llm_with_image_context = mllm.bind(images=image_b64)
    res = llm_with_image_context.invoke(question)
    return res


# create mmodel
mllm = Ollama(model=st.session_state["model"])

with st.chat_message("question"):#, avatar="🧑‍🚀"):
    st.markdown(f"**{question}**", unsafe_allow_html=True)
with st.spinner("Thinking..."):
    res = run_llm(question, image_b64, model=st.session_state["model"])
    with st.chat_message("response"):#, avatar="🤖"):
        st.write(res)

