import os
import openai
import streamlit as st
from PIL import Image
from io import BytesIO
import requests

openai.api_key = os.getenv('OPENAI_API_KEY')


def translate_to_english(text):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=f"{text}\nEnglish translation,output in a one word:",
      temperature=0.5,
      max_tokens=60
    )
    return response.choices[0].text.strip()

def generate_characteristic_words(input_text):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=f"{input_text}\nFrom the input text, output exactly two characteristic nouns in English, separated by a comma.",
      temperature=0.5,
      max_tokens=60
    )
    words = response.choices[0].text.strip().split(', ')
    return words if len(words) == 2 else [words[0], ""]

st.title("絵日記AI")

word_d_options = ["boy", "girl", "dog", "cat", "dolphin", "dragon", "giraffe"]
word_d = st.selectbox("Please select a subject:", word_d_options)

word_a_options = ["red", "blue", "yellow", "black", "white", "purple"]
word_a = st.selectbox("今日の気分を色で表すと？", word_a_options)

word_b_options = [
    "今日一番ワクワクした瞬間は何でしたか？",  # thrilling
    "今日最も驚いた出来事は何でしたか？",  # amazing
    "今日最も困った瞬間は何でしたか？",  # trouble
    "今日あなたを最も笑わせたことは何でしたか？"  # smiling
]
word_emo = {
    "今日一番ワクワクした瞬間は何でしたか？": "thrilling",
    "今日最も驚いた出来事は何でしたか？": "amazing",
    "今日最も困った瞬間は何でしたか？": "trouble",
    "今日あなたを最も笑わせたことは何でしたか？": "funny"
}
word_b = st.selectbox("Please select a question:", word_b_options)
word_b_response = st.text_input(word_b, "")

if word_a and word_b_response:
    generate_button = st.button("絵を生成")
    if generate_button:
        words_bc = generate_characteristic_words(word_b_response)
        st.write("word_a:", word_a)
        st.write("word_b:", words_bc[0])
        if words_bc[1]:
            st.write("word_c:", words_bc[1])
        else:
            st.write("word_c: None")

        # Assuming the image creation function is available
        response = openai.Image.create(
            prompt=f"Comical and cartoonish {word_d} with a {word_a} atmosphere, showing {word_emo[word_b]}, with {words_bc[0]} and {words_bc[1]}",
            n=1,
            size="1024x1024"
        )

        # assuming 'data' field in response containing 'url' in its first index
        image_url = response['data'][0]['url']

        # Fetching the image using the given url
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        # Display the fetched image
        st.image(img, caption='Generated Image', use_column_width=True)

        # Download the image
        img.save("generated_image.png")
        st.download_button(
            "Download the image",
            data=open("generated_image.png", "rb"),
            file_name="generated_image.png",
            mime="image/png",
        )
