## Write a streamlit application that can translate english to french, or help me optimising my french

import streamlit as st
import pandas as pd
import openai
import numpy as np
import time
import random
import regex as re
from datetime import datetime
import os
import random

openai.api_type         = os.getenv("OPENAI_TYPE")
openai.api_base         = os.getenv("OPENAI_BASE")
openai.api_version      = os.getenv("OPENAI_VERSION")
openai.api_key          = os.getenv("OPENAI_KEY")



        

if 'alphabet' not in st.session_state:
    st.session_state.alphabet= {0: 'a',
        1: 'b',
        2: 'c',
        3: 'd',
        4: 'e',
        5: 'f',
        6: 'g',
        7: 'h',
        8: 'i',
        9: 'j',
        10: 'k',
        11: 'l',
        12: 'm',
        13: 'n',
        14: 'o',
        15: 'p',
        16: 'q',
        17: 'r',
        18: 's',
        19: 't',
        20: 'u',
        21: 'v',
        22: 'w',
        23: 'x',
        24: 'y',
        25: 'z'}
    
if 'letter_x' not in st.session_state:
    st.session_state.letter_x = "a"
if 'letter_y' not in st.session_state:
    st.session_state.letter_y = "b"
    
if 'int_x' not in st.session_state:
    st.session_state.int_x = 0
if 'int_x' not in st.session_state:
    st.session_state.int_x = 0

def letter_x():
    st.session_state.letter_x = random.choice(list(st.session_state.alphabet.values()))
    int_x()
    
def letter_y():
    while st.session_state.letter_y == st.session_state.letter_x:
        st.session_state.letter_y = random.choice(list(st.session_state.alphabet.values()))
    
    int_y()    
    
def int_x():
    for k in st.session_state.alphabet.keys():
        if st.session_state.alphabet[k] == st.session_state.letter_x:
            st.session_state.int_x = k
           
           
def int_y():
    for k in st.session_state.alphabet.keys():
        if st.session_state.alphabet[k] == st.session_state.letter_y:
            st.session_state.int_y = k

st.header("Alphabet Test")
letter_y()
letter_x()

st.write(f"Does letter {st.session_state.letter_x} come before letter {st.session_state.letter_y} in the alphabet?")
if st.button("Yes"):
    if st.session_state.int_x < st.session_state.int_y:
        st.write("Correct!")
    else:
        st.write("Wrong!")
    
    letter_x()
    letter_y()
if st.button("No"):
    x = 0
    if  st.session_state.int_x > st.session_state.int_y:
        st.write("Correct!")
    else:
        st.write("Wrong!")
    letter_x()
    letter_y()