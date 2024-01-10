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
from ruamel.yaml.compat import StringIO
from PIL import Image

st.title("Story Generator")
st.sidebar.header("Story Generator")

backend = ml_backend()
oa = openai
oa.api_key = os.getenv("OPENAI_API_KEY")

#### START SESSION VARS
if 'entitytype' not in st.session_state:
    st.session_state.entitytype = 'ENTITY: '
if 'elementmodel' not in st.session_state:
    st.session_state.elementmodel = 'unknown'
    #elementlist = ["earth", "water", "fire", "air"]
    #mymodel = st.selectbox('Select an element: ', elementlist)
if 'gentype' not in st.session_state:
    st.session_state.gentype = 'awaiting gentype...'
if 'intro' not in st.session_state:
    st.session_state.intro = 'awaiting intro...'

if 'entitybio' not in st.session_state:
    st.session_state.entitybio = 'awaiting entitybio...'
if 'chartloc' not in st.session_state:
    st.session_state.chartloc = 'chart_not_ready.png'

if 'prompt0' not in st.session_state:
    st.session_state.prompt0 = 'awaiting act0 prompt...'
if 'prompt1' not in st.session_state:
    st.session_state.prompt1 = 'awaiting act1 prompt...'
if 'prompt2' not in st.session_state:
    st.session_state.prompt2 = 'awaiting act2 prompt...'
if 'prompt3' not in st.session_state:
    st.session_state.prompt3 = 'awaiting act3 prompt...'

if 'ouput1' not in st.session_state:
    st.session_state.ouput1 = 'awaiting output1...'
if 'ouput2' not in st.session_state:
    st.session_state.ouput2 = 'awaiting output2...'
if 'ouput3' not in st.session_state:
    st.session_state.ouput3 = 'awaiting output3...'
#### END SESSION VARS

with st.form(key="init"):
    #### ENTITY CONFIG DROPDOWN
    # folder path
    dir_path = 'data'
    # list to store files
    res = ["----"]
    # Iterate directory
    for file in os.listdir(dir_path):
        # check only text files
        if file.endswith('.yaml'):
            res.append(file)


    configfile = st.selectbox(
        'Select your config file ',
        res)
    st.write('You selected:', configfile)
    load_yaml_data = st.form_submit_button(label='Load Config Data')

    ###################

# placeholder econfig
econfig={
    'entitydescr':{
        "bio": st.session_state.entitybio,
        "type": st.session_state.entitytype,
        "element": st.session_state.elementmodel,
        "id": st.session_state.gentype
    } ,
    'entitydata': {
        "chartstorage": st.session_state.chartloc,
    },
    'prompt': {
        "intro": st.session_state.intro,
        "act0descr": st.session_state.prompt0,
        "act1descr": st.session_state.prompt1,
        "act2descr": st.session_state.prompt2,
        "act3descr": st.session_state.prompt3,
    },
    'storygenerations': {}
}

if load_yaml_data:

    #### LOAD SELECTED ENTITY CONFIG
    yaml = YAML(typ='safe')
    yaml.default_flow_style = False
    configfile = "./data/" + configfile

    with open(configfile, encoding='utf-8') as f:
        econfig = yaml.load(f)
    #### END CONFIG

### DEFINE BASIC PARAMS
maxlength = 300
#maxlength = 60
gencount = 1
selectedmodel = "unknown"
try:
   st.session_state.entity = st.session_state.entitytype.upper()
except:
    st.session_state.entity = "Entity not detected"
###################

### READ YAML DATA INTO SESSION STATE

try:
    st.session_state.entitybio = econfig['entitydescr']['bio']
except: st.session_state.entitybio = "not set"

try:
    st.session_state.entitytype = econfig['entitydescr']['type']
except:
    st.session_state.entitytype = "entitytype not defined in yaml"

try:
    st.session_state.elementmodel = econfig['entitydescr']['element']
except:
    st.session_state.elementmodel = "elementmodel not defined in yaml"

try:
    st.session_state.gentype = econfig['entitydescr']['id']
except:
    st.session_state.gentype = "gentype/id not defined in yaml"

try:
    st.session_state.chartloc = econfig['entitydata']['chartstorage']
except:
    st.session_state.chartloc = "chart_not_ready.png"

try:
    st.session_state.intro = econfig['prompt']['intro']
except:
    st.session_state.intro =  "intro not defined in yaml"

try:
    st.session_state.prompt0 = econfig['prompt']['act0descr']
except:
    st.session_state.prompt0 = "prompt0/act0 not defined in yaml"

try:
    st.session_state.prompt1 = econfig['prompt']['act1descr']
except:
    st.session_state.prompt1 = "prompt1/act1 not defined in yaml"

try:
    st.session_state.prompt2 = econfig['prompt']['act2descr']
except:
    st.session_state.prompt2 = "prompt2/act2 not defined in yaml"
try:
    st.session_state.prompt3 = econfig['prompt']['act3descr']
except:
    st.session_state.prompt3 = "prompt3/act3 not defined in yaml"
###################
st.write('Configured element:', st.session_state.elementmodel)

### DEFINE ML MODEL
#print("checking model: " + st.session_state.elementmodel)
if st.session_state.elementmodel == "earth":
    selectedmodel = "davinci:ft-personal-2022-05-08-13-37-54"
elif st.session_state.elementmodel == "water":
    selectedmodel = "davinci:ft-personal:water-2022-03-31-23-56-04"
elif st.session_state.elementmodel == "fire":
    selectedmodel = "davinci:ft-personal:fire-2022-07-06-02-12-31"
elif st.session_state.elementmodel == "air":
    selectedmodel = "davinci:ft-personal:air-2022-07-05-23-19-23"
else:
    selectedmodel = "unknown"
    print("Selected model is unknown:" + st.session_state.elementmodel)
###################



### SET CHART LOC
try:
    image = Image.open("charts/" + st.session_state.chartloc)
    st.image(image, caption='Data Chart')
except:
    image = Image.open("charts/chart_not_ready.png")
    st.image(image, caption='Data Chart')
###################

with st.form(key="form"):
    output1 = '---'

    try:
        introact0rawprompt = st.session_state.intro + st.session_state.prompt0  + st.session_state.entitybio + '\\n\\n'
        introact0prettyprompt = st.session_state.intro.replace('\\n', '\n')  + st.session_state.prompt0.replace('\\n', '\n') + st.session_state.entitybio + '\n\n'
    except:
        introact0rawprompt = "Trouble building prompt due to null values"
        introact0prettyprompt = "Trouble building prompt due to null values"

    act1rawprompt = st.session_state.prompt1
    act1prettyprompt = st.session_state.prompt1.replace('\\n', ' \n')

    introact0prompt = st.text_area('Intro', introact0prettyprompt, height=300)
    act1prompt = st.text_area('Act 1', act1prettyprompt, height=150)

    finalrawprompt1 = introact0rawprompt + act1rawprompt
    finalprettyprompt1 = introact0prettyprompt + act1prettyprompt

    submit_act1 = st.form_submit_button(label='Generate Act1')
    if submit_act1:
        with st.spinner("Generating Act..."):
            #output1 = backend.generate_text_test1(finalrawprompt1)
            output1 = backend.generate_text(finalrawprompt1,maxlength,selectedmodel)
        st.session_state.ouput1 = output1.replace('\n','\\n')
        #st.session_state.intro = introact0prompt.replace('\n','\\n')
        st.session_state.prompt1 = act1prompt.replace('\n','\\n')
        #st.write('Output1 = ', st.session_state.ouput1)
        st.session_state.entity = st.session_state.entity.upper()

    act1static = st.session_state.entity + ': some output...'
    act1res = st.markdown(st.session_state.entity + ': ' + st.session_state.ouput1)

with st.form(key="form1a"):
    edit_ouput1 = st.text_area('Output 1', st.session_state.ouput1, height=150)
    save_ouput1 = st.form_submit_button(label='Save Output 1')
    if save_ouput1:
        st.session_state.ouput1 = edit_ouput1

with st.form(key="form2"):
    output2 = '---'
    try:
       act2rawprompt = finalrawprompt1 + st.session_state.ouput1 + '\\n\\n' + st.session_state.prompt2 + '\\n\\n'
       act2prettyprompt = finalprettyprompt1  + '\n\n' + st.session_state.ouput1.replace('\\n', '\n') + '\n\n' + st.session_state.prompt2.replace('\\n', '\n') + '\n\n'
    except:
       act2rawprompt = "Trouble building prompt due to null values"
       act2prettyprompt = "Trouble building prompt due to null values"

    try:
       act2only = st.text_area('Act 2', st.session_state.prompt2.replace('\\n', '\n'), height=150)
    except:
       act2only = st.text_area('Act 2', "Trouble building prompt due to null values", height=150)

    submit_act2 = st.form_submit_button(label='Generate Act2')
    if submit_act2:
        with st.spinner("Generating Act..."):
            #output2 = backend.generate_text_test2(act2rawprompt)
            output2 = backend.generate_text(act2rawprompt, maxlength, selectedmodel)
        st.session_state.ouput2 = output2
        st.session_state.prompt2 = act2only.replace('\n', '\\n')
        #st.write('Output2 = ', st.session_state.ouput2)

    act2res = st.markdown(st.session_state.entity + ': ' + st.session_state.ouput2)

with st.form(key="form2a"):
    edit_ouput2 = st.text_area('Output 2', st.session_state.ouput2, height=150)
    save_ouput2 = st.form_submit_button(label='Save Output 2')
    if save_ouput2:
        st.session_state.ouput2 = edit_ouput2

with st.form(key="form3"):
    output3 = '---'
    try:
        act3rawprompt = act2rawprompt  + st.session_state.ouput2 + '\\n\\n' +  st.session_state.prompt3
        act3prettyprompt = act2prettyprompt  + '\n\n' + st.session_state.ouput2.replace('\\n', '\n') + '\n\n' +  st.session_state.prompt3.replace('\\n', '\n')
    except:
        act3rawprompt = "Trouble building prompt due to null values"
        act3prettyprompt = "Trouble building prompt due to null values"

    try:
        act3only = st.text_area('Act 3', st.session_state.prompt3.replace('\\n', '\n'), height=150)
    except:
        act3only = st.text_area('Act 3', "Trouble building prompt due to null values", height=150)

    submit_act3 = st.form_submit_button(label='Generate Act3')
    #act3res = st.text(act3static)
    if submit_act3:
        with st.spinner("Generating Act..."):
            #output3 = backend.generate_text_test3(act3rawprompt)
            output3 = backend.generate_text(act3rawprompt, maxlength, selectedmodel)
        st.session_state.ouput3 = output3
        st.session_state.prompt3 = act3only.replace('\n', '\\n')
        #st.write('Output3 = ', st.session_state.ouput3)

    act2res = st.markdown(st.session_state.entity + ': ' + st.session_state.ouput3)

with st.form(key="form3a"):
    edit_ouput3 = st.text_area('Output 3', st.session_state.ouput3, height=150)
    save_ouput3 = st.form_submit_button(label='Save Output 3')
    if save_ouput3:
        st.session_state.ouput3 = edit_ouput3

with st.form(key="form4"):
    show_story = st.form_submit_button(label='Show final story')
    if show_story:
        st.markdown(st.session_state.ouput1)
        st.markdown(st.session_state.ouput2)
        st.markdown(st.session_state.ouput3)

st.markdown("""---""")
st.markdown("### Debug Options")

with st.form(key="form5"):
    show_prompt = st.form_submit_button(label='Show raw prompts only')
    if show_prompt:
        st.markdown(st.session_state.prompt1)
        st.markdown(st.session_state.prompt2)
        st.markdown(st.session_state.prompt3)

with st.form(key="form6"):
    show_rawprompts = st.form_submit_button(label='Show raw prompt including ouput')
    if show_rawprompts:
        st.markdown(act3rawprompt)

with st.form(key="form7"):
    show_prettyprompts = st.form_submit_button(label='Show pretty prompt including ouput')
    if show_prettyprompts:
        act3prettyrompt = act3rawprompt.replace('\\n', '\n\n') + st.session_state.ouput3
        st.markdown(act3prettyrompt)

with st.form(key="form8"):
    # datetime object containing current date and time
    dt_string = datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
    ###### WRITE GENERATIONS TO YAML
    genid = dt_string
    act1gen = st.session_state.ouput1.replace('\\n','\n')
    act2gen = st.session_state.ouput2.replace('\\n','\n')
    act3gen = st.session_state.ouput3.replace('\\n','\n')
    genpayload = [{
        'aagen_id': genid,
        'act1gen': act1gen.strip(),
        'act2gen': act2gen.strip(),
        'act3gen': act3gen.strip()
                  }]
    econfig['storygenerations']= genpayload
    show_yamlstory = st.form_submit_button(label='Show yaml generation')
    if show_yamlstory:
        class MyYAML(YAML):
            def dump(self, data, stream=None, **kw):
                inefficient = False
                if stream is None:
                    inefficient = True
                    stream = StringIO()
                YAML.dump(self, data, stream, **kw)
                if inefficient:
                    return stream.getvalue()


        myaml = MyYAML()
        yaml = YAML()
        def tr(s):
            return s.replace('\n', '<br>')
        st.markdown(myaml.dump(econfig,transform=tr))
        yaml.dump(econfig,sys.stdout)
