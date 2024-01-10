import streamlit as st
import openai
from ml_backend import ml_backend
import os
import re
import openai
import datetime
from contextlib import redirect_stdout
import sys
from ruamel.yaml import YAML
from PIL import Image

st.set_page_config(page_title="Manual Story Generator", page_icon="📖")
st.markdown("# Manual Story Generator")
st.sidebar.header("Manual Story Generator")

backend = ml_backend()
oa = openai
oa.api_key = os.getenv("OPENAI_API_KEY")

#### START SESSION VARS
if 'man_entitytype' not in st.session_state:
    st.session_state.man_entitytype = 'ENTITY: '
if 'man_elementmodel' not in st.session_state:
    st.session_state.man_elementmodel = 'awaiting element model'
if 'man_gentype' not in st.session_state:
    st.session_state.man_gentype = 'awaiting gentype...'
if 'man_intro' not in st.session_state:
    st.session_state.man_intro = 'The following play reveals the inner monologue of a <descriptor> <type>. It is divided into several acts. Throughout these acts, the <type> describes its inner and outer transformation:\n\nThe first act starts like this:\n\n'

if 'man_entitybio' not in st.session_state:
    st.session_state.man_entitybio = 'I am a <descriptor> <type>...'
if 'man_chartloc' not in st.session_state:
    st.session_state.man_chartloc = 'chart_not_ready.png'

if 'man_prompt0' not in st.session_state:
    st.session_state.man_prompt0 = 'ACT0: The <measurementdescr> is not yet known. The <type> introduces itself and describes its surroundings.\nENTITY: <entitybio>'
if 'man_prompt1' not in st.session_state:
    st.session_state.man_prompt1 = 'ACT1: <add prompt>\nENTITY:'
if 'man_prompt2' not in st.session_state:
    st.session_state.man_prompt2 = 'ACT2: <add prompt>\nENTITY:'
if 'man_prompt3' not in st.session_state:
    st.session_state.man_prompt3 = 'ACT3: <add prompt>\nENTITY:'

if 'man_ouput1' not in st.session_state:
    st.session_state.man_ouput1 = 'awaiting output1...'
if 'man_ouput2' not in st.session_state:
    st.session_state.man_ouput2 = 'awaiting output2...'
if 'man_ouput3' not in st.session_state:
    st.session_state.man_ouput3 = 'awaiting output3...'

econfig={
    'entitydescr':{
        "bio": st.session_state.man_entitybio,
        "type": st.session_state.man_entitytype,
        "element": st.session_state.man_elementmodel,
        "id": st.session_state.man_gentype
    } ,
    'entitydata': {
        "chartstorage": st.session_state.man_chartloc,
    },
    'prompt': {
        "intro": st.session_state.man_intro,
        "act0descr": st.session_state.man_prompt0,
        "act1descr": st.session_state.man_prompt1,
        "act2descr": st.session_state.man_prompt2,
        "act3descr": st.session_state.man_prompt3,
    }
}

maxlength = 300
#maxlength = 60
gencount = 1
selectedmodel = "unknown"
try:
   st.session_state.man_entity = st.session_state.man_entitytype.upper()
except:
    st.session_state.man_entity = "Entity not detected"
###################

### READ YAML DATA INTO SESSION STATE

try:
    st.session_state.man_entitybio = econfig['entitydescr']['bio']
except: st.session_state.man_entitybio = "not set"

try:
    st.session_state.man_entitytype = econfig['entitydescr']['type']
except:
    st.session_state.man_entitytype = "entitytype not defined in yaml"

try:
    st.session_state.man_elementmodel = econfig['entitydescr']['element']
except:
    st.session_state.man_elementmodel = "elementmodel not defined in yaml"

try:
    st.session_state.man_gentype = econfig['entitydescr']['id']
except:
    st.session_state.man_gentype = "gentype/id not defined in yaml"

try:
    st.session_state.man_chartloc = econfig['entitydata']['chartstorage']
except:
    st.session_state.man_chartloc = "chart_not_ready.png"

try:
    st.session_state.man_intro = econfig['prompt']['intro']
except:
    st.session_state.man_intro =  "intro not defined in yaml"

try:
    st.session_state.man_prompt0 = econfig['prompt']['act0descr']
except:
    st.session_state.man_prompt0 = "prompt0/act0 not defined in yaml"

try:
    st.session_state.man_prompt1 = econfig['prompt']['act1descr']
except:
    st.session_state.man_prompt1 = "prompt1/act1 not defined in yaml"

try:
    st.session_state.man_prompt2 = econfig['prompt']['act2descr']
except:
    st.session_state.man_prompt2 = "prompt2/act2 not defined in yaml"
try:
    st.session_state.man_prompt3 = econfig['prompt']['act3descr']
except:
    st.session_state.man_prompt3 = "prompt3/act3 not defined in yaml"

with st.form(key="form"):
    output1 = ''

    elementlist = ["---","earth", "water", "fire", "air"]
    mymodel = st.selectbox('Select an element: ', elementlist)
    st.session_state.man_elementmodel = mymodel
    if mymodel in ["earth", "water", "fire", "air"]:
        ### DEFINE ML MODEL
        # print("checking model: " + st.session_state.man_elementmodel)
        if st.session_state.man_elementmodel == "earth":
            selectedmodel = "davinci:ft-personal-2022-05-08-13-37-54"
        elif st.session_state.man_elementmodel == "water":
            selectedmodel = "davinci:ft-personal:water-2022-03-31-23-56-04"
        elif st.session_state.man_elementmodel == "fire":
            selectedmodel = "davinci:ft-personal:fire-2022-07-06-02-12-31"
        elif st.session_state.man_elementmodel == "air":
            selectedmodel = "davinci:ft-personal:air-2022-07-05-23-19-23"
        else:
            selectedmodel = "unknown"
            print("Selected model is unknown:" + st.session_state.man_elementmodel)
            st.write('Configured element:', st.session_state.man_elementmodel)
        ###################

    try:
        introact0rawprompt = st.session_state.man_intro + st.session_state.man_prompt0  + st.session_state.man_entitybio + '\\n\\n'
        introact0prettyprompt = st.session_state.man_intro.replace('\\n', '\n')  + st.session_state.man_prompt0.replace('\\n', '\n') + st.session_state.man_entitybio + '\n\n'
    except:
        introact0rawprompt = "Trouble building prompt due to null values"
        introact0prettyprompt = "Trouble building prompt due to null values"

    act1rawprompt = st.session_state.man_prompt1
    act1prettyprompt = st.session_state.man_prompt1.replace('\\n', ' \n')

    introact0prompt = st.text_area('Intro', introact0prettyprompt, height=300)
    act1prompt = st.text_area('Act 1', act1prettyprompt, height=150)

    finalrawprompt1 = introact0rawprompt + act1rawprompt
    finalprettyprompt1 = introact0prettyprompt + act1prettyprompt

    submit_act1 = st.form_submit_button(label='Generate Act1')
    if submit_act1:
        st.write("Requesting with model type:", st.session_state.man_elementmodel)
        st.write("Requesting with model:", selectedmodel)
        with st.spinner("Generating Act..."):
            #output1 = backend.generate_text_test1(finalrawprompt1)
            output1 = backend.generate_text(finalrawprompt1,maxlength,selectedmodel)
        st.session_state.man_ouput1 = output1.replace('\n','\\n')
        #st.session_state.man_intro = introact0prompt.replace('\n','\\n')
        st.session_state.man_prompt1 = act1prompt.replace('\n','\\n')
        #st.write('Output1 = ', st.session_state.man_ouput1)
        st.session_state.man_entity = st.session_state.man_entity.upper()

    act1static = st.session_state.man_entity + ': some output...'
    act1res = st.markdown(st.session_state.man_entity + ': ' + st.session_state.man_ouput1)

with st.form(key="form1a"):
    edit_ouput1 = st.text_area('Output 1', st.session_state.man_ouput1, height=150)
    save_ouput1 = st.form_submit_button(label='Save Output 1')
    if save_ouput1:
        st.session_state.man_ouput1 = edit_ouput1

with st.form(key="form2"):
    output2 = ''
    try:
       act2rawprompt = finalrawprompt1 + st.session_state.man_ouput1 + '\\n\\n' + st.session_state.man_prompt2 + '\\n\\n'
       act2prettyprompt = finalprettyprompt1  + '\n\n' + st.session_state.man_ouput1.replace('\\n', '\n') + '\n\n' + st.session_state.man_prompt2.replace('\\n', '\n') + '\n\n'
    except:
       act2rawprompt = "Trouble building prompt due to null values"
       act2prettyprompt = "Trouble building prompt due to null values"

    try:
       act2only = st.text_area('Act 2', st.session_state.man_prompt2.replace('\\n', '\n'), height=150)
    except:
       act2only = st.text_area('Act 2', "Trouble building prompt due to null values", height=150)

    submit_act2 = st.form_submit_button(label='Generate Act2')
    if submit_act2:
        with st.spinner("Generating Act..."):
            #output2 = backend.generate_text_test2(act2rawprompt)
            output2 = backend.generate_text(act2rawprompt, maxlength, selectedmodel)
        st.session_state.man_ouput2 = output2
        st.session_state.man_prompt2 = act2only.replace('\n', '\\n')
        #st.write('Output2 = ', st.session_state.man_ouput2)

    act2res = st.markdown(st.session_state.man_entity + ': ' + st.session_state.man_ouput2)

with st.form(key="form2a"):
    edit_ouput2 = st.text_area('Output 2', st.session_state.man_ouput2, height=150)
    save_ouput2 = st.form_submit_button(label='Save Output 2')
    if save_ouput2:
        st.session_state.man_ouput2 = edit_ouput2

with st.form(key="form3"):
    output3 = ''
    try:
        act3rawprompt = act2rawprompt  + st.session_state.man_ouput2 + '\\n\\n' +  st.session_state.man_prompt3
        act3prettyprompt = act2prettyprompt  + '\n\n' + st.session_state.man_ouput2.replace('\\n', '\n') + '\n\n' +  st.session_state.man_prompt3.replace('\\n', '\n')
    except:
        act3rawprompt = "Trouble building prompt due to null values"
        act3prettyprompt = "Trouble building prompt due to null values"

    try:
        act3only = st.text_area('Act 3', st.session_state.man_prompt3.replace('\\n', '\n'), height=150)
    except:
        act3only = st.text_area('Act 3', "Trouble building prompt due to null values", height=150)

    submit_act3 = st.form_submit_button(label='Generate Act3')
    #act3res = st.text(act3static)
    if submit_act3:
        with st.spinner("Generating Act..."):
            #output3 = backend.generate_text_test3(act3rawprompt)
            output3 = backend.generate_text(act3rawprompt, maxlength, selectedmodel)
        st.session_state.man_ouput3 = output3
        st.session_state.man_prompt3 = act3only.replace('\n', '\\n')
        #st.write('Output3 = ', st.session_state.man_ouput3)

    act2res = st.markdown(st.session_state.man_entity + ': ' + st.session_state.man_ouput3)

with st.form(key="form3a"):
    edit_ouput3 = st.text_area('Output 3', st.session_state.man_ouput3, height=150)
    save_ouput3 = st.form_submit_button(label='Save Output 3')
    if save_ouput3:
        st.session_state.man_ouput3 = edit_ouput3

with st.form(key="form4"):
    show_story = st.form_submit_button(label='Show final story')
    if show_story:
        st.markdown(st.session_state.man_ouput1)
        st.markdown(st.session_state.man_ouput2)
        st.markdown(st.session_state.man_ouput3)

st.markdown("""---""")
st.markdown("### Debug Options")

with st.form(key="form5"):
    show_prompt = st.form_submit_button(label='Show raw prompts only')
    if show_prompt:
        st.markdown(st.session_state.man_prompt1)
        st.markdown(st.session_state.man_prompt2)
        st.markdown(st.session_state.man_prompt3)

with st.form(key="form6"):
    show_rawprompts = st.form_submit_button(label='Show raw prompt including ouput')
    if show_rawprompts:
        st.markdown(act3rawprompt)

with st.form(key="form7"):
    show_prettyprompts = st.form_submit_button(label='Show pretty prompt including ouput')
    if show_prettyprompts:
        act3prettyrompt = act3rawprompt.replace('\\n', '\n\n') + st.session_state.man_ouput3
        st.markdown(act3prettyrompt)