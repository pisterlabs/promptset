# application that transforms description to SVG code

import streamlit as st
import openai
from backend import backend_app



# title of the app
st.title("SVG Code Generator")
st.text("Code written by: Yara Abazah")

# description of the app
st.markdown("""

# About
## This app generates SVG code from a description given by the user. 
## The app uses the GPT-3 engine to generate the SVG code. 
## The app is built using Streamlit and the backend is written in Python.

""")


st.markdown("## Description")

# new instance of the backend
backend = backend_app()

with st.form(key="form"):
    description = st.text_area(label="Describe what you want to transform into SVG code")
    st.text(f"(Example: a red circle with black outline)")
    # start = st.text_area(label="Enter start of SVG code here")
    start = "<svg height"
    # end = st.text_area(label="Enter end of SVG code here")
    end = "</svg>"
    submit = st.form_submit_button(label="Generate SVG Code")

    # if the user clicks the submit button
    if submit:
        with st.spinner("Generating SVG Code..."):
           # generate the SVG code 
           output = backend.generate_svg("Transform the following description into SVG code: "+ description, start, end)
        
        # display the SVG code
        st.markdown("# SVG Code")
        st.subheader(start+output+end)

        # display the SVG code as an image using the SVG code and transform it to html code
        st.markdown("# SVG Image")
        # add html code to display the SVG code as an image
        svg_code = f"<img src='data:image/svg+xml;utf8,{start+output+end}' width='400'>"
        st.markdown(svg_code, unsafe_allow_html=True)

        
        # st.image(start+output+end, width=400)
        # st.image(start+output+end, width=400)

        # st.code(svg_code, language="html")
        # # display the SVG code as an SVG image
        # st.markdown(f"## SVG Image")
        # st.image(svg_code, width=400)
        # 

