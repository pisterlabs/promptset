import streamlit as st
from openai import OpenAI 
from PIL import Image
from tempfile import NamedTemporaryFile
from config import pagesetup as ps
import cv2
import os
from functions import image_analyze as imanalyze, image_encode as imencode, image_frame_description as imframedesc


# SESSION STATE
if "vidtempfile" not in st.session_state:
    st.session_state.vidtempfile = None
if "vidtempfilepath" not in st.session_state:
    st.session_state.vidtempfilepath = ""
if "vidtotalframes" not in st.session_state:
    st.session_state.vidtotalframes = 0
if "vidcurrentframe" not in st.session_state:
    st.session_state.vidcurrentframe = 0
if "weightclasses" not in st.session_state:
    st.session_state.weightclasses = [125, 133, 141, 149, 157, 165, 174, 184, 197, 285]
if "rankingsallrecords" not in st.session_state:
    st.session_state.rankingsallrecords = []
    
#0. Page Config
st.set_page_config("WrestleAI", initial_sidebar_state="collapsed", layout="wide")

ps.set_title("WrestleAI", "Home")
#ps.set_page_overview("Overview", "**FEOC Assistant** provides a way to quickly ask about the FEOC")
container1 = st.container()
with container1:
    cc = st.columns([0.4,0.1,0.5])
    with cc[2]:
        ps.set_page_overview_no_div("Welcome to WrestleAI", "Integrating AI in Wrestling for Enhanced Training and Performance.")
        
        st.divider()


        cc1 = st.columns(2)
        with cc1[0]:
            st.markdown("**Athletes**")
            st.markdown("""```
WrestleAI provides a complete suite of tools - both 
on and off the mat - personalized for each athlete.
                    """)
            st.markdown("**Any Style**")
            st.markdown("""```
WrestleAI can provide the same rigour of training
across Freestyle, Greco, and Folkstyle.
                    """)

        with cc1[1]:
            st.markdown("**Coaches**")
            st.markdown("""```
WrestleAI provides brand new set of tools that
remove tedius tasks allowing coaches to coach.
                    """)
            st.markdown("**Any Level**")
            st.markdown("""```
WrestleAI meets you at your level and
adapts from kids club through olympic-levels.
                    """)
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("""
<div style='text-align: center;'>
    <span style='font-size: larger;'>Discover how WrestleAI transforms wrestling training with AI-driven analytics and real-time feedback.</span>
</div>
""", unsafe_allow_html=True)
        st.link_button("Sign-Up Today", "https://flowgenius.com",use_container_width=True)

    with cc[0]:
        image = Image.open("assets/logo.png")
        st.image(image)
    
st.divider()
container2=st.container()
ps.set_page_overview_no_div("Capabilities", "Highlighting the AI integration in wrestling training and coaching.")
st.write("")
st.write("")
cc3=st.columns(4)
with cc3[0]:
    container4=st.container()
    with container4:
        st.markdown("#### **Technique Breakdown**")
        e4=st.expander("", expanded=True)
        with e4:
            st.write("Learn technique.")
            st.button("View", key="btne4", type="primary")
    container5=st.container()
    with container5:
        st.markdown("#### **Match Strategy Advisor**")
        e5= st.expander("", expanded=True)
        with e5:
            st.write("Create a match strategy.")
            st.button("View", key="btne5", type="primary")
with cc3[1]:
    container6=st.container()
    with container6:
        st.markdown("#### **Training Program Generator**")
        e6=st.expander("", expanded=True)
        with e6:
            st.write("Create custom training programs.")
            st.button("View", key="btne6", type="primary")
    container7=st.container()
    with container7:
        st.markdown("#### **Historical Match Analysis**")
        e7= st.expander("", expanded=True)
        with e7:
            st.write("Analyze historical matches.")
            st.button("View", key="btne7", type="primary")
with cc3[2]:
    container8=st.container()
    with container8:
        st.markdown("#### **Injury Prevention Tips**")
        e8=st.expander("", expanded=True)
        with e8:
            st.write("Get personalized injury prevention.")
            st.button("View", key="btne8", type="primary")
    container9=st.container()
    with container9:
        st.markdown("#### **Nutrition Guide**")
        e9= st.expander("", expanded=True)
        with e9:
            st.write("View rule updates and changes.")
            st.button("View", key="btne9", type="primary")
with cc3[3]:
    container10=st.container()
    with container10:
        st.markdown("#### **Competition Preparedness Checklist**")
        e10= st.expander("", expanded=True)
        with e10:
            st.write("View rule updates and changes.")
            st.button("View", key="btne10", type="primary")
    container11=st.container()
    with container11:
        st.markdown("#### **Rule Change Updates**")
        e11=st.expander("", expanded=True)
        with e11:
            st.write("View rule updates and changes.")
            st.button("View", key="btne11", type="primary")
    #with cc[2]:
        #ps.set_page_overview_no_div("Welcome to WrestleAI", "WrestleAI is cutting edge, one of a kind technology.")
        #st.divider()
        #st.markdown("**Purchasers**")
        #st.markdown("""```
                #By choosing to back reduced emission products, 
#you set a commendable standard. Every purchase you make 
#takes us one step closer to a cleaner, better world.
#                  """)