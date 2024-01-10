import streamlit as st
from datetime import datetime
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


compliance_template = "./complianceTemplates/anyTaskCompliance.txt"

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


if check_password():
    def complianceChecker(_media):
        with open(compliance_template) as comp_temp:
            prefix = f'''
            You are a compliance expert at a top 100 company and are tasked with making sure the given media is compliant with company policy. 
            here is the copy of the company policy: {comp_temp.readlines()}.
            
            check and verify that the following media is compliant. please make sure if the letters are properly capitalized
            Media:{_media}'''

        return prefix



    def generate_output(prompt, temperature=1, model="gpt-4-0613"):
        """Generate output using the OpenAI API."""
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                n=1,
                stop=None,
                temperature=temperature
            )
            message = response.choices[0].message.content
            return message
        except Exception as e:
            st.error(f"Error: {e}")
            return None

    def main(_media):
        complianceReturned = complianceChecker(_media)
        holder = generate_output(complianceReturned)
        
        print(f"Bot {datetime.now()}: "+holder)
        return holder
        # return holder

        



    


    CompliantHolder = ["Explore thousands of Tasks at AnyTask.com",
            "Explore thousands of Sellers at AnyTask.com",
            "AnyTask.com is a freelance platform.",
            "This is AnyTask.com",
            "AnyTask.com"] 
    NotCompliantHolder = ["Explore thousands of tasks at anytask",
                        "Explore thousands of sellers at anytask",
                        "anytask is a marketplace.",
                        "We are Anytask.com",
                        "Anytask.com"]






    st.title("Compliance Bot")
    input_text = st.text_area("Enter the media here:", height=300)
    print(f"User {datetime.now()}: "+input_text)
    if st.button("Submit"):
        result = main(input_text)
        st.markdown(f"**Result:**\n{result}")

# st.file_uploader()