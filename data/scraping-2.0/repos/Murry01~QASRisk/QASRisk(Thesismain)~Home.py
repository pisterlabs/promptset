import time
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import openai

st.set_page_config(
    page_title="My App",
    page_icon=":house:",
)

st.subheader(
    "Development of Question Answering System Based on Knowledge Graph and Generative Pre-Trained Transformer to Improve Risk Identification in Construction Projects"
)
# st.header("Q & A System for Construction Risk Management")
# st.write(
#     "##### :blue[Question answering system for Identifying Risks in Tunnel Projects]"
# )
st.info(
    """This is a Web-based Question-Answering System (QAS) powered by Knowledge Graph (KG) and 
        Generative Pre-trained Transformer (GPT) model to aid the Risk Identification in Construction Projects.
        The Question-Answering System for Construction Risk (QASRisk) allows users to interact with a Risk Knowledge Graph (RisKG) 
        using the OpenAi' GPT-3 model. This system can be used to answer questions about Construction Project Risks.
 """
)
img = Image.open("image1.jpeg")
st.image("image1.jpeg")


c1, c2, c3 = st.columns(3)
with c1:
    st.info(
        "**Google scholar: [Murry](https://scholar.google.com/citations?user=qinOzG0AAAAJ&hl=en)**",
        icon="💡",
    )
with c2:
    st.info("**GitHub: [Murry01](https://github.com/Murry01/)**", icon="💻")
with c3:
    st.info(
        "**LinekedIn: [Muritala](https://www.linkedin.com/in/muritala-adebayo-isah-1656b768/)**",
        icon="🧠",
    )

# Footer
st.markdown("---")
