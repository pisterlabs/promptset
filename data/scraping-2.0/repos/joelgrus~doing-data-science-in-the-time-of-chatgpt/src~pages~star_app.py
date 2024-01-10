import streamlit as st
import openai
import os

# Replace 'your_openai_api_key' with your actual OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]


def predict_stars(text):
    response = openai.Completion.create(
        engine="text-davinci-003",  # or use the latest model available
        prompt=f"""The following text is a review. 
Please judge how many stars the review is, from one to five.
Where 1-star is a terrible review and 5-star is a glowing review.
Please return your answer as a string containing between 1 and 5 star emoji (⭐).
Here is the review:

{text}""",
    )
    return response.choices[0].text.strip()


# Streamlit UI
st.title("Review Star Predictor")

st.link_button("Some reviews", "https://www.yelp.com/biz/marys-tacos-helotes")

with st.form("text_analysis_form"):
    text_input = st.text_area("Enter the review:", height=200)
    submit_button = st.form_submit_button("Predict Stars")

if submit_button and text_input:
    with st.spinner("Analyzing..."):
        stars = predict_stars(text_input)

    st.subheader("Star Prediction")
    st.write(stars)

else:
    st.write("Enter some text and click predict to get started!")

# To run the Streamlit app, save this code in a file named app.py and run `streamlit run app.py`
