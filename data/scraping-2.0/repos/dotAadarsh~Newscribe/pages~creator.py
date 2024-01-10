import streamlit as st
import openai

openai.api_key = st.secrets["OPEN_AI_API"]

with st.sidebar:
    st.image("./assets/news.png")
    st.success("Newscribe is an AI-powered app that transcribes news videos into written blog posts, simplifying content creation for bloggers and news writers.")
    st.info("For this demo, I have integrated the FirstPost news channels youtube content")
    st.warning("Feedbacks are welcome. Connect with me on [Twitter](https://twitter.com/DotAadarsh)")

@st.cache_data
def text_generate(prompt, token):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.7,
    max_tokens=token,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    generated_text = response["choices"][0]["text"]
    return generated_text

@st.cache_data
def generateImage(imagePrompt, size):
    response = openai.Image.create(
    prompt= imagePrompt,
    n=1,
    size=size
    )
    image_url = response['data'][0]['url']

    return image_url


def main():
    st.header("Content Studio")
    st.caption("Create | Edit | Share")

    option = st.selectbox("What do you want to create?", ["Blog Post", "Thumbnail", "Tweet"], index=2)

    if option == "Blog Post":
        input_prompt = st.text_input("About what?")
        if input_prompt:
            prompt = "Create a blog post in markdown format for " + input_prompt
            blog_post = text_generate(prompt, 250)
            st.write(blog_post)

    elif option == "Thumbnail":
        imagePrompt = st.text_input("Type the image prompt here")
        if imagePrompt:
            size = st.radio("select the size of the image", ('256x256', '512x512', '1024x1024'))

            st.image(generateImage(imagePrompt, size))

    elif option == "Tweet": 
        input_prompt = st.text_input("Whats happening?")
        if input_prompt:
            prompt = "Write a tweet on " + input_prompt
            tweet = text_generate(prompt, 100)
            st.write(tweet)


if __name__ == '__main__':
    main()