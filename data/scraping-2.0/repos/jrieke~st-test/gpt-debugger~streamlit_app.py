import traceback
from contextlib import contextmanager
from inspect import currentframe, getframeinfo

import openai
import streamlit as st

prompt_template = """
I run the following code with Streamlit:

{}

It shows me the following exception message:

{}

Can you write a 2-3 sentences long description of what the issue is and show how to fix the code. Only print out the lines that changed. You can use markdown formatting for the code. Also say in the filename and line that needs to be changed. 
"""

openai.api_key = st.secrets["openai"]


@st.cache_data(show_spinner="Asking GPT for advice...")
def gpt(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].text


@contextmanager
def debug():

    # Set everything up before yield to extract the code in the contextmanager.
    # See https://stackoverflow.com/a/73788472
    cf = currentframe()
    first_line = cf.f_back.f_back.f_lineno
    filename = getframeinfo(cf.f_back.f_back).filename

    try:
        yield
    except Exception as e:

        # Show the exception in the app (just like Streamlit would do without the
        # contextmanager).
        st.write(e)

        # Now extract the code in the contextmanager.
        cf = currentframe()
        last_line = cf.f_back.f_back.f_lineno
        with open(filename) as f:
            lines = f.readlines()[first_line:last_line]
        code = "".join(lines).rstrip()

        # st.code(code)

        # Get the exception message and stack trace.
        exception_text = str(e) + "\n" + traceback.format_exc()

        # Enter code + exception into the template prompt and send to GPT.
        prompt = prompt_template.format(code, exception_text)
        result = gpt(prompt)

        # Show the answer in an info box.
        st.info(result, icon="🤖")


with debug():
    import streamlit as st

    st.title("Streamlit + GPT for debugging 🐛")

    slider_value = st.slider("My slider", 0, 100, 50)
    if st.button("Add a bug"):
        st.write(slider_value + "foo")
