import streamlit as st
from ml_backend import ml_backend
import os
import openai
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO
from PIL import Image
import pandas as pd

st.title("Story Generator")
st.sidebar.header("Story Generator")

backend = ml_backend()
oa = openai
oa.api_key = os.getenv("OPENAI_API_KEY")

###################
# placeholder econfig
econfig = {
    'entitydescr': {
        "bio": "nothing",
        "type": "nothing",
        "element": "nothing",
        "id": "unknown"
    },
    'entitydata': {
        "chartstorage": "nothing",
    },
    'prompt': {
        "intro": "nothing",
        "act0descr": "nothing",
        "act1descr": "nothing",
        "act2descr": "nothing",
        "act3descr": "nothing",
    },
    'storygenerations': {}
}
eid = 'unknown'

# Initialization
if 'loadstate' not in st.session_state:
    st.session_state['loadstate'] = 'init'
if 'act1state' not in st.session_state:
    st.session_state['act1state'] = 'init'
if 'act2state' not in st.session_state:
    st.session_state['act2state'] = 'init'
if 'act3state' not in st.session_state:
    st.session_state['act3state'] = 'init'
if 'storygencounter' not in st.session_state:
    st.session_state['storygencounter'] = 0
if 'act1cleaned' not in st.session_state:
    st.session_state['act1cleaned'] = ''
if 'act2cleaned' not in st.session_state:
    st.session_state['act2cleaned'] = ''
if 'act3scleaned' not in st.session_state:
    st.session_state['act3scleaned'] = ''
if 'modeltypeindex' not in st.session_state:
    st.session_state['modeltypeindex'] = 0
if 'gpt4persona' not in st.session_state:
    st.session_state['gpt4continuation'] = "The following text is play about "

### Persona: EARTH:
if 'gpt4persona' not in st.session_state:
    st.session_state['gpt4persona'] = 'You are an advanced literary AI that can perfectly imitate any literary style. Your output is an amalgamation of all the styles that you have been trained on. Right now, you have been trained on the following works: "A season in Hell " by Arthur Rimbaud, "Illuminations" by Arthur Rimbaud, all poems by Audre Lorde, "Erotism: Death and Sensuality" by George Bataille, "Body Art and Performance: The Body as Language" by Lea Vergine, "Nadja" by André Breton, "Tropic of Cancer" by Henry Miller, and "Mad love" by André Breton. Any time I ask you continue a text, you will continue the text in a style that is based on an almagamation of: A) the styles that you have been trained on and B) the style found in the original text. You will never mention the names of any of these authors in your output. You will also never mention the titles of the works that you have been trained on.'

### Persona: AIR: 'You are an advanced literary AI that can perfectly imitate any literary style. Your output is an amalgamation of all the styles that you have been trained on. Right now, you have been trained on the following works: "The Passion according to G.H." by Clarice Lispector, "Spell of the Sensuous" by David Abram, all poems by Marina Tsvetaeva, "The Myth of Sisyphus" by Albert Camus, "Sonnets to Orpheus " by Rainer Maria Rilke, "Duino Elegies" by Rainer Maria Rilke, "A Lover''s Discourse: Fragments" by Roland Barthes, "The Metamorphoses" by Ovid,  "Songs of Innocence and of Experience" by William Blake, "The Marriage of Heaven and Hell" by William Blake and "I love Dick" by Chris Kraus. Any time I ask you continue a text, you will continue the text in a style that is based on an almagamation of: A) the styles that you have been trained on and B) the style found in the original text. You will never mention the names of any of these authors in your output. You will also never mention the titles of the works that you have been trained on. Finally, you will never mention the word "C02"'

### Persona: WATER:'You are an advanced literary AI that can perfectly imitate any literary style. Your output is an amalgamation of all the styles that you have been trained on. Right now, you have been trained on the following works: "Agua Viva" by Clarice Lispector, "The Temptation to Exist" by Emil Cioran, "The Book of Disquiet" by Fernando Pessoa, "À rebours" by J. K. Huysmans, "Voyage of Time" by Terrence Malick, "The hour of the Star" by Clarice Lispector. Any time I ask you continue a text, you will continue the text in a style that is based on an almagamation of: A) the styles that you have been trained on and B)the style found in the original text. You will never mention the names of any of these authors in your output. You will also never mention the titles of the works that you have been trained on. Finally, you will never mention the word "precipitation"'

### Persona: FIRE: 'You are an advanced literary AI that can perfectly imitate any literary style. Your output is an amalgamation of all the styles that you have been trained on. Right now, you have been trained on the following works: "A season in Hell " by Arthur Rimbaud, "Illuminations" by Arthur Rimbaud, all poems by Audre Lorde, "Erotism: Death and Sensuality" by George Bataille, "Body Art and Performance: The Body as Language" by Lea Vergine, "Nadja" by André Breton, "Tropic of Cancer" by Henry Miller, and "Mad love" by André Breton. Any time I ask you continue a text, you will continue the text in a style that is based on an almagamation of: A) the styles that you have been trained on and B) the style found in the original text. You will never mention the names of any of these authors in your output. You will also never mention the titles of the works that you have been trained on.'

### Persona: EARTH: 'You are an advanced literary AI that can perfectly imitate any literary style. Your output is an amalgamation of all the styles that you have been trained on. Right now, you have been trained on the following works: "Underland: A Deep Time Journey" by Robert Macfarlane, "Poetic Intention" by Édouard Glissant, "Last Temptation of Christ" by Nikos Kazantzakis, and all Poems by Sylvia Plath. Any time I ask you continue a text, you will continue the text in a style that is based on an almagamation of: A) the styles that you have been trained on and B) the style found in the original text. You will never mention the names of any of these authors in your output. You will also never mention the titles of the works that you have been trained on.'

### Define Utility functions
# Uses st.experimental_memo to only rerun when the query changes or after 5 mins.
#@st.experimental_memo(ttl=300)
def load_csv_as_df(file):
    #df = pd.read_csv(file,index_col=0)
    df = pd.read_csv(file)
    return (df)
def write_to_csv(df,index,value):

    pd.to_csv(df)
# def load_sheet_as_df(url, sajson, id, act):
#     gc = gs.service_account(filename=sajson)
#     sh = gc.open_by_url(url)
#     ws = sh.worksheet('data')
#     df = pd.DataFrame(ws.get_all_records())
#     df.set_index("field", inplace=True)
#     print(f'Loading {act} data for {id}')
#
#     return (ws, df)

def flatten_yaml(theyaml):
    # Flatten YAML, separate keys with "_"
    df2 = pd.json_normalize(theyaml, sep='_')

    # Transpose Yaml so that keys are in 1 column
    df2_tr = df2.T
    dft = df2_tr.reset_index()
    return dft

def write_to_sheet(ws, findex, fvalue):
    ws.update_cell(findex, 2, fvalue)
    # st.write(f'wrote value {fvalue} to row {findex}')

def load_chart(chartstorage):
    ### Try to load the appropriate chart image from
    imgpath = "charts/" + str(chartstorage)
    try:
        image = Image.open(imgpath)
    except:
        image = Image.open("charts/chart_not_ready.png")
        st.write(f'{imgpath} does not appear to exist')
        # st.image(image, caption='Data Chart')
    ###################
    return image

def get_model(entitydescr_element):
    ### Try to identify the correct ML model to use from sheet-db
    ### DEFINE ML MODEL
    # print("checking model: " + st.session_state.elementmodel)
    # yel = df2_tr.loc['entitydescr_element']
    yelv = entitydescr_element
    # print(yel)
    #yelv = yel[0]
    if yelv == "earth":
        selectedmodel = "davinci:ft-personal-2022-05-08-13-37-54"
    elif yelv == "water":
        selectedmodel = "davinci:ft-personal:water-2022-03-31-23-56-04"
    elif yelv == "fire":
        selectedmodel = "davinci:ft-personal:fire-2022-07-06-02-12-31"
    elif yelv == "air":
        selectedmodel = "davinci:ft-personal:air-2022-07-05-23-19-23"
    else:
        selectedmodel = "unknown"

    return selectedmodel
    ###################

## Define the function that will get the correct row number of a keyvalue (for updating)

def get_index(df, search):
    result = df.loc[df['field'] == search]
    findex = result.index.tolist()[0]
    return findex

def get_value(df, search):
    res = df.loc[df['field'] == search]
    rind = res.index.tolist()[0]
    output = res.at[rind, 'value']

    return output

with st.form(key="init"):

    #### ENTITY CONFIG DROPDOWN
    # folder path
    dir_path = 'data'
    # list to store files
    res = ["----"]
    # Iterate directory
    for file in os.listdir(dir_path):
        # check only yaml files
        if file.endswith('.yaml'):
            res.append(file)
    res = sorted(res)
    configfile = st.selectbox(
        'Select your config file ',
        res)
    st.markdown(f'You selected: :green[{configfile}]')
    modeltype = st.radio("Select the model type",('GTP3 trained on selected texts', 'GPT4 imitating style of listed texts'),st.session_state['modeltypeindex'])

    load_yaml_data = st.form_submit_button(label='Load Config Data')

    # If 'Load Config Data' Button was pressed
    if load_yaml_data:
        if modeltype == 'GPT4 imitating style of listed texts':
            st.session_state['modeltypeindex'] = 1
        st.session_state['storygencounter'] = 0
        #### LOAD SELECTED ENTITY CONFIG FROM YAML INTO DF2
        yaml = YAML(typ='safe')
        yaml.default_flow_style = False
        configfile = "./data/" + configfile

        try:
            ### Try to load and flatten the selected YAML file
            with open(configfile, encoding='utf-8') as f:
                econfig = yaml.load(f)
            df2_tr = flatten_yaml(econfig)
            df2_tr.columns = ['field', 'value']
            #### Write yaml to CSV

            df2_tr.to_csv("streamlit_db.csv")
            #### Load CSV again
            #df = load_csv_as_df("streamlit_db.csv")
            df = load_csv_as_df("streamlit_db.csv")

            try:
                get_value(df, 'storygenerations_aagen_id')
            except:
                extragen = pd.DataFrame({'field': ['storygenerations_aagen_id','storygenerations_act1gen', 'storygenerations_act2gen', 'storygenerations_act3gen'],
                                         'value': [get_value(df,'entitydescr_id'), 'tbd','tbd', 'tbd']})
                df = pd.concat([df, extragen], ignore_index=True, sort=False)


            st.write(f'Loading Complete')
            st.write(st.session_state.loadstate)
            st.write(modeltype)
            st.session_state['loadstate'] = 'loaded'
            st.session_state['act1state'] = 'ready'
            st.session_state['act2state'] = 'reload'
            st.session_state['act3state'] = 'reload'

        ### Stop if the Yaml file does not load
        # (or perhaps a problem with flatten, sheetwrite, modelselect, or chartselect
        except Exception as e:
            st.write(f'Sorry, something funky going on with the file "{configfile}".')
            st.write(f'Inform Merlin of the issue and try another file.')
            st.write(f'The issue was:\n\n{e}')
            print(e)
            st.stop()

        df.to_csv("streamlit_db.csv")

with st.form(key="act0_1"):
    if st.session_state['act1state'] == 'ready':
        output1 = '---'
        #### Reload sheet in case new stuff was written
        df = load_csv_as_df("streamlit_db.csv")

        modelname = get_value(df,'entitydescr_element')
        selectedmodel = get_model(modelname)

        yid = get_value(df,'entitydescr_id')

        st.markdown(f'Data loaded for: :green[{yid}]')

        modeldisplay = selectedmodel.replace(':', '_')
        st.markdown(f"**Tone of voice type**: :green[{get_value(df,'entitydescr_element')}]")
        st.markdown(f"**Selectd ML model**: :green[{modeldisplay}]")

        image = load_chart(get_value(df,'entitydata_chartstorage'))
        st.image(image, caption='Data Chart')

        try:
            # Build the first prompt with intro, act 0 + bio (encoded line breaks)
            introact0rawprompt = get_value(df,'prompt_intro') + get_value(df,'prompt_act0descr') + get_value(df,'entitydescr_bio') + '\\n\\n'
            # Build the first prompt with intro, act 0 + bio (visible line breaks)
            introact0prettyprompt = get_value(df,'prompt_intro').replace('\\n', '\n')  + get_value(df,'prompt_act0descr').replace('\\n', '\n') + get_value(df,'entitydescr_bio') + '\n\n'
        except Exception as e:
            introact0rawprompt = "Trouble building prompt due to: " + str(e)
            introact0prettyprompt = "Trouble building prompt due to: " + str(e)

        # Build the act 1 prompt (encoded line breaks)
        act1rawprompt = get_value(df,'prompt_act1descr')
        # Build the act 1 prompt (visible line breaks)
        act1prettyprompt =get_value(df,'prompt_act1descr').replace('\\n', ' \n')

        # Render the max tokens and prompts into fields
        maxlength = st.text_input('Max Number of words to generate:', 325,  max_chars=3)
        introact0prompt = st.text_area('Intro', introact0prettyprompt, height=300)
        act1prompt = st.text_area('Act 1', act1prettyprompt, height=150)

        # Build the combined intro and act 1 prompt (encoded line breaks)
        finalrawprompt1 = introact0rawprompt + act1rawprompt
        # Build the combined intro and act 1 prompt (visible line breaks)
        finalprettyprompt1 = introact0prettyprompt + act1prettyprompt

        submit_act1 = st.form_submit_button(label='Generate Act1')

        # if the submit button is pressed, send the whole prompt to OpenAI
        if submit_act1:
            with st.spinner("Generating Act..."):
                if modeltype == 'GPT4 imitating style of listed texts':
                    oa.api_key = os.getenv("OPENAI_API_KEY_MC")
                    output1 = backend.gengpt4_text(finalrawprompt1, maxlength, st.session_state['gpt4persona'])
                else:
                    output1 = backend.generate_text(finalrawprompt1, maxlength, selectedmodel)
                print(output1 )
            st.success('Done!')

            # Write the revised prompt for Act 1 to the sheet-db
            writeindex = get_index(df, 'prompt_act1descr')
            #write_to_sheet(ws,writeindex, act1prompt.replace('\n','\\n'))
            df.at[writeindex, 'value'] = act1prompt.replace('\n','\\n')

            # Write the output generation for Act 1 to the sheet-db
            writeindex = get_index(df,'storygenerations_act1gen')
            #write_to_sheet(ws,writeindex, output1.replace('\n','\\n'))
            df.at[writeindex, 'value'] = output1.replace('\n','\\n')

            # Write the id of what the generation is for to the sheet-db
            writeindex = get_index(df,'storygenerations_aagen_id')
            #write_to_sheet(ws,writeindex, yid)
            df.at[writeindex, 'value'] = yid

            # Write the uppercase version of the entity name (i.e. 'LAND:') to the sheet-db
            writeindex = get_index(df,'entitydescr_type')
            #write_to_sheet(ws,writeindex, df.loc['entitydescr_type'][0].upper())
            df.at[writeindex, 'value'] = get_value(df,'entitydescr_type').upper()

            act1static = get_value(df,'entitydescr_type').upper() + ': some output...'
            act1res = st.markdown(get_value(df,'entitydescr_type').upper() + ': ' + output1)

            # Write everything we have done to the CSV
            df.to_csv("streamlit_db.csv")

with st.form(key="act0_1_out"):
    if st.session_state['act1state'] == 'ready':
        #### Reload sheet in case new stuff was written
        df = load_csv_as_df("streamlit_db.csv")

        yid = get_value(df,'storygenerations_aagen_id')
        st.markdown(f'Most recent output for: :green[{yid}]')

        modelname = get_value(df,'entitydescr_element')
        selectedmodel = get_model(modelname)
        outputsize = len(str(get_value(df,'storygenerations_act1gen'))) / 3
        #outputsize = len(str(st.session_state['act1cleaned'])) / 3
        fieldheight = int(outputsize)
        edit_ouput1 = st.text_area('Output 1', get_value(df,'storygenerations_act1gen'), height=fieldheight)
        #edit_ouput1 = st.text_area('Output 1', st.session_state['act1cleaned'], height=fieldheight)
        clean_act1 = st.form_submit_button(label='Try to clean the response')
        save_ouput1 = st.form_submit_button(label='Save Output 1')
        proceed1 = st.form_submit_button(label='Proceed to Act 2')

        if clean_act1:
            with st.spinner("Trying to clean the text..."):
                cloutput1 = backend.clean_response(get_value(df,'storygenerations_act1gen'))
                st.session_state['act1cleaned'] = cloutput1
            st.success('Done!')
            st.markdown(':green[Cleaned version ]')
            st.markdown(f':green[{cloutput1}]')

        if save_ouput1:
            with st.spinner("Saving Updated Output For Act 1......"):
                writeindex = get_index(df,'storygenerations_act1gen')
                df.at[writeindex, 'value'] = edit_ouput1

            # Write everything we have done to the CSV
            df.to_csv("streamlit_db.csv")
            st.success('Saved Updated Output For Act 1!')
        if proceed1:
            st.session_state['act2state'] = 'ready'

with st.form(key="act_2"):
    if st.session_state['act2state'] == 'ready':
        output1 = '---'
        #### Reload sheet in case new stuff was written ########################################################
        df = load_csv_as_df("streamlit_db.csv")

        modelname = get_value(df,'entitydescr_element')
        selectedmodel = get_model(modelname)

        try:
            # Build the first prompt with intro, act 0 + bio (encoded line breaks) + act 1 + act1out + act2
            act1rawprompt = get_value(df,'prompt_intro') + get_value(df,'prompt_act0descr') + get_value(df,'entitydescr_bio') + '\\n\\n' +get_value(df,'prompt_act1descr') + '\\n' + get_value(df,'storygenerations_act1gen') + '\\n\\n'
            # Build the first prompt with intro, act 0 + bio (visible line breaks)
            act1prettyprompt = get_value(df,'prompt_intro').replace('\\n', '\n') + get_value(df,'prompt_act0descr').replace('\\n', '\n') + get_value(df,'entitydescr_bio') + '\n\n'+ get_value(df,'prompt_act1descr') + '\n' + get_value(df,'storygenerations_act1gen') + '\n\n'
        except Exception as e:
            act1rawprompt = "Trouble building prompt due to: " + str(e)
            act1prettyprompt = "Trouble building prompt due to: " + str(e)

        # Build the act 2 prompt (encoded line breaks)
        act2rawprompt = get_value(df,'prompt_act2descr')
        # Build the act 2 prompt (visible line breaks)
        act2prettyprompt = get_value(df,'prompt_act2descr').replace('\\n', ' \n')

        # Render the max tokens and prompts into fields
        maxlength = st.text_input('Max Number of words to generate:', 325,  max_chars=3)
        act2prompt = st.text_area('Act 2', act2prettyprompt, height=150)

        # Build the combined intro and act 1 prompt (encoded line breaks)
        finalrawprompt2 = act1rawprompt + act2rawprompt
        # Build the combined intro and act 1 prompt (visible line breaks)
        finalprettyprompt2 = act1prettyprompt + act2prettyprompt

        submit_act2 = st.form_submit_button(label='Generate Act2')

        # if the submit button is pressed, send the whole prompt to OpenAI
        if submit_act2:
            with st.spinner("Generating Act..."):
                if modeltype == 'GPT4 imitating style of listed texts':
                    oa.api_key = os.getenv("OPENAI_API_KEY_MC")
                    output2 = backend.gengpt4_text(finalrawprompt2,maxlength,selectedmodel)
                else:
                    output2 = backend.generate_text(finalrawprompt2,maxlength,selectedmodel)
                #print(output2 )
            st.success('Done!')

            # Write the revised prompt for Act 2 to the sheet-db
            writeindex = get_index(df,'prompt_act2descr')
            df.at[writeindex, 'value'] = act2prompt.replace('\n','\\n')

            # Write the output generation for Act 1 to the sheet-db
            writeindex = get_index(df,'storygenerations_act2gen')
            df.at[writeindex, 'value'] = output2.replace('\n','\\n')

            # Write the uppercase version of the entity name (i.e. 'LAND:') to the sheet-db
            writeindex = get_index(df,'entitydescr_type')
            df.at[writeindex, 'value'] =get_value(df,'entitydescr_type').upper()

            act2static = get_value(df,'entitydescr_type').upper() + ': some output...'
            act2res = st.markdown(get_value(df,'entitydescr_type').upper() + ': ' + output2)

            # Write everything we have done to the CSV
            df.to_csv("streamlit_db.csv")

with st.form(key="act_2_out"):
    if st.session_state['act2state'] == 'ready':
        #### Reload sheet in case new stuff was written ########################################################
        df = load_csv_as_df("streamlit_db.csv")

        yid = get_value(df,'storygenerations_aagen_id')
        st.markdown(f'Most recent output for: :green[{yid}]')

        modelname = get_value(df,'entitydescr_element')
        selectedmodel = get_model(modelname)
        outputsize = len(str(get_value(df,'storygenerations_act2gen'))) / 3
        fieldheight = int(outputsize)
        edit_ouput2 = st.text_area('Output 2', get_value(df,'storygenerations_act2gen'), height=fieldheight)
        clean_act2 = st.form_submit_button(label='Try to clean the response')
        save_ouput2 = st.form_submit_button(label='Save Output 2')
        proceed2 = st.form_submit_button(label='Proceed to Act 3')

        if clean_act2:
            with st.spinner("Trying to clean the text..."):
                cloutput2 = backend.clean_response(get_value(df,'storygenerations_act2gen'))
                st.session_state['act2cleaned'] = cloutput2
            st.success('Done!')
            st.markdown(':green[Cleaned version ]')
            st.markdown(f':green[{cloutput2}]')

        if save_ouput2:
            with st.spinner("Saving Updated Output For Act 2..."):
                writeindex = get_index(df,'storygenerations_act2gen')
                df.at[writeindex, 'value'] = edit_ouput2

            # Write everything we have done to the CSV
            df.to_csv("streamlit_db.csv")
            st.success('Saved Updated Output For Act 2!')
        if proceed2:
            st.session_state['act3state'] = 'ready'

with st.form(key="act_3"):
    if st.session_state['act3state'] == 'ready':
        output1 = '---'
        #### Reload sheet in case new stuff was written ########################################################
        df = load_csv_as_df("streamlit_db.csv")

        modelname = get_value(df,'entitydescr_element')
        selectedmodel = get_model(modelname)

        try:
            # Build the first prompt with intro, act 0 + bio (encoded line breaks) + act 1 + act1out + act 2 + act2out + act3
            act2rawprompt = get_value(df,'prompt_intro') + get_value(df,'prompt_act0descr') + get_value(df,'entitydescr_bio') + '\\n\\n' + get_value(df,'prompt_act1descr') + '\\n' + get_value(df,'storygenerations_act1gen') + get_value(df,'prompt_act2descr') + '\\n' + get_value(df,'storygenerations_act2gen') + '\\n\\n'
            # Build the first prompt with intro, act 0 + bio (visible line breaks)
            act2prettyprompt = get_value(df,'prompt_intro').replace('\\n', '\n') + get_value(df,'prompt_act0descr').replace('\\n', '\n') + get_value(df,'entitydescr_bio') + '\n\n'+ get_value(df,'prompt_act1descr') + '\n' + get_value(df,'storygenerations_act1gen')+ '\n\n'+ get_value(df,'prompt_act2descr') + '\n' + get_value(df,'storygenerations_act2gen') + '\n\n'
        except Exception as e:
            act2rawprompt = "Trouble building prompt due to: " + str(e)
            act2prettyprompt = "Trouble building prompt due to: " + str(e)

        # Build the act 3 prompt (encoded line breaks)
        act3rawprompt = get_value(df,'prompt_act3descr')
        # Build the act 3 prompt (visible line breaks)
        act3prettyprompt = get_value(df,'prompt_act3descr').replace('\\n', ' \n')

        # Render the max tokens and prompts into fields
        maxlength = st.text_input('Max Number of words to generate:', 325,  max_chars=3)
        act3prompt = st.text_area('Act 3', act3prettyprompt, height=150)

        # Build the combined intro and act 1 prompt (encoded line breaks)
        finalrawprompt3 = act2rawprompt + act3rawprompt
        # Build the combined intro and act 1 prompt (visible line breaks)
        finalprettyprompt3 = act2prettyprompt + act3prettyprompt

        submit_act3 = st.form_submit_button(label='Generate Act3')

        # if the submit button is pressed, send the whole prompt to OpenAI
        if submit_act3:
            with st.spinner("Generating Act 3..."):
                if modeltype == 'GPT4 imitating style of listed texts':
                    oa.api_key = os.getenv("OPENAI_API_KEY_MC")
                    output3 = backend.gengpt4_text(finalrawprompt2, maxlength, selectedmodel)
                else:
                    output3 = backend.generate_text(finalrawprompt2, maxlength, selectedmodel)
                #print(output2 )
            st.success('Done!')

            # Write the revised prompt for Act 3 to the sheet-db
            writeindex = get_index(df,'prompt_act3descr')
            df.at[writeindex, 'value'] = act3prompt.replace('\n','\\n')

            # Write the output generation for Act 3to the sheet-db
            writeindex = get_index(df,'storygenerations_act3gen')
            df.at[writeindex, 'value'] = output3.replace('\n','\\n')

            act2static = get_value(df,'entitydescr_type').upper() + ': some output...'
            act2res = st.markdown(get_value(df,'entitydescr_type').upper() + ': ' + output3)

            # Write everything we have done to the CSV
            df.to_csv("streamlit_db.csv")

with st.form(key="act_3_out"):
    if st.session_state['act3state'] == 'ready':
        #### Reload sheet in case new stuff was written ########################################################
        df = load_csv_as_df("streamlit_db.csv")

        yid = get_value(df,'storygenerations_aagen_id')
        st.markdown(f'Most recent output for: :green[{yid}]')

        modelname = get_value(df,'entitydescr_element')
        selectedmodel = get_model(modelname)
        outputsize = len(str(get_value(df,'storygenerations_act3gen'))) / 3
        fieldheight = int(outputsize)
        edit_ouput3 = st.text_area('Output 3', get_value(df,'storygenerations_act3gen'), height=fieldheight)
        clean_act3 = st.form_submit_button(label='Try to clean the response')
        save_ouput3 = st.form_submit_button(label='Save Output 3')
        proceed3 = st.form_submit_button(label='Proceed to Whole Story')

        if clean_act3:
            with st.spinner("Trying to clean the text..."):
                cloutput3 = backend.clean_response(get_value(df,'storygenerations_act3gen'))
                st.session_state['act3cleaned'] = cloutput3
            st.success('Done!')
            st.markdown(':green[Cleaned version ]')
            st.markdown(f':green[{cloutput3}]')

        if save_ouput3:
            with st.spinner("Saving Updated Output For Act 3..."):
                writeindex = get_index(df,'storygenerations_act3gen')
                df.at[writeindex, 'value'] = edit_ouput3

            # Write everything we have done to the CSV
            df.to_csv("streamlit_db.csv")
            st.success('Saved Updated Output For Act 3!')

        if proceed3:
            st.session_state['act3state'] = 'complete'
            st.write(f"State is {st.session_state['act3state'] }")

        # Write everything we have done to the CSV
        df.to_csv("streamlit_db.csv")

with st.form(key="whole_story"):
    if st.session_state['act3state'] == 'complete':
        #### Reload sheet in case new stuff was written ########################################################
        df = load_csv_as_df("streamlit_db.csv")

        st.markdown("You can find these generations on the [Output App page](output_app)")

        st.markdown(get_value(df,'storygenerations_act1gen').replace('\\n','\n'))
        st.markdown(get_value(df,'storygenerations_act2gen').replace('\\n','\n'))
        st.markdown(get_value(df,'storygenerations_act3gen').replace('\\n','\n'))

        saveit = st.form_submit_button(label='Save to generations for this session')
        if saveit:
            gen = st.session_state['storygencounter']
            act1label = 'storygenerations_act1gen_' + str(gen)
            act2label = 'storygenerations_act2gen_' + str(gen)
            act3label = 'storygenerations_act3gen_' + str(gen)
            act1row = [act1label, get_value(df,'storygenerations_act1gen').replace('\\n','\n')]
            act2row = [act2label, get_value(df,'storygenerations_act2gen').replace('\\n','\n')]
            act3row = [act3label, get_value(df,'storygenerations_act3gen').replace('\\n','\n')]
            extragen = pd.DataFrame({'field': [act1label, act2label, act3label],
             'value': [act1row[1], act2row[1], act3row[1]]})
            df2 = pd.concat([df, extragen], ignore_index=True, sort=False)
            df2.to_csv('streamlit_db.csv')
            st.session_state['storygencounter'] = gen + 1
            st.write(f"Generation #{gen} Saved")
