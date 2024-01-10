import streamlit as st
import openai
from ml_backend import ml_backend

st.title("Automatic Email Generator App")
st.text("by Team Phoenix")
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://p1.pxfuel.com/preview/879/495/832/writing-pen-books-book.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

st.markdown(""" 

  

## Business Benefits and Usecases:
* Time saved writing medium-long sized emails
* Mental Energy is conserved
* Anxiety of writing a **professional sounding** email (or email with any writing style) is removed as the GPT3 Language model used is trained from a variety of many different internet sources

""")

st.markdown("# Generate Email")

backend = ml_backend()

with st.form(key="form"):
    prompt = st.text_input("Describe the Kind of Email you want to be written.")
    st.text(f"(Example: Write me a professional sounding email to my boss)")

    start = st.text_input("Begin writing the first few or several words of your email:")

    slider = st.slider("How many characters do you want your email to be? ", min_value=64, max_value=750)
    st.text("(A typical email is usually 100-500 characters)")

    submit_button = st.form_submit_button(label='Generate Email')

    if submit_button:
        with st.spinner("Generating Email..."):
            output = backend.generate_email(prompt, start)
        st.markdown("# Email Output:")
        st.subheader(start + output)

        st.markdown("____")
        st.markdown("# Send Your Email")
        st.subheader("You can press the Generate Email Button again if you're unhappy with the model's output")
        
        st.subheader("Otherwise:")
        st.text(output)
        url = "https://mail.google.com/mail/?view=cm&fs=1&to=&su=&body=" + backend.replace_spaces_with_pluses(start + output)

        st.markdown("[Click me to send the email]({})".format(url))
