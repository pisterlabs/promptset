import streamlit as st
from langchain.llms import OpenAI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import joblib
from PIL import Image
import sklearn
import pyPRISM
import holoviews as hv
hv.extension('bokeh')
from pyPRISMfile import *
# Config function
img=Image.open("nano.jpg")
st.set_page_config(page_title='Nanoparticles distribution prediction',page_icon=img)

# hide main menu and footer
hide_menu_style= """
    <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style,unsafe_allow_html=True)

def user_input_features():
    ponp=st.sidebar.slider('Interaction between polymers and nanoparticles: ',0.0,2.5, 0.4)
    npnp=st.sidebar.slider('Interaction between nanoparticles and nanoparticles: ',0.0,2.5, 0.4)
    d=st.sidebar.slider('Diameter of nanoparticles: ',1,10,4)
    phi=st.sidebar.slider('1000*Phi: ',1,5,2)
    cLength=st.sidebar.slider('Length of polymer chain: ',5,60,10)
    st.sidebar.subheader('Distance Range')
    distance_str_min = st.sidebar.text_input('Minimum distance in nm: ','0.075')
    distance_str_width_range = st.sidebar.text_input('Width range in nm: ','150')
    distance_min= float(distance_str_min)
    distance_range= float(distance_str_width_range)
    Po_NP=pd.DataFrame({'Po_NP':[ponp]*2000})
    NP_NP=pd.DataFrame({'NP_NP':[npnp]*2000})
    D_aim=pd.DataFrame({'D_aim':[d]*2000})
    Phi=pd.DataFrame({'Phi':[phi/1000]*2000})
    Chain_length=pd.DataFrame({'Chain_length':[cLength]*2000})
    distance = pd.DataFrame({'distance': np.linspace(distance_min, distance_min + distance_range, 2000)})
    features=pd.concat([Po_NP,NP_NP,D_aim,Phi,Chain_length,distance], axis=1)
    features.columns = ['Po_NP','NP_NP','D_aim','Phi','Chain length','distance']
    #features=data
    return features


#try:
    #model = joblib.load("/mount/src/app/model.pkl")
#except Exception as e:
    #st.error(f"An error occurred while loading the model: {e}")

st.markdown("""
<style>
    [data-testid=stSidebar][aria-expanded="true"]{
        background-color: #ee9322;
        border-radius: 20px;
    }
</style>
""",unsafe_allow_html=True)

#letter-spacing: 4px;
#text-shadow: 3px 1px blue;
with st.sidebar:
    st.write("""
        <style>
            .st-eb {
                background-color: #EE9322; /* Background color */
                border-radius: 10px; /* Border radius */
            }

            .st-ec .st-ed { /* Thumb of the slider */
                background-color: #FF0000; /* Thumb color */
            }
        </style>
    """, unsafe_allow_html=True)
    title = '<p style="font-family: Courier;text-align: center;font-weight: bolder; color: Darkblue; font-size: 30px;">Input parameters</p>'
    st.markdown(title, unsafe_allow_html=True)
        
        # Use HTML to create a div with background color, opacity, and rounded border
    info_box = """
    <div style="background-color: #BCD5ED;text-align: center; padding: 10px;border-radius:10px">
        <p>Please enter inputs for the calculation.</p>
    </div>
    """
    st.markdown(info_box, unsafe_allow_html=True)
df = user_input_features()
title_main='<h1 style="text-align:center; font-weight: bolder;color: #EE9322;text-shadow: 3px 1px blue;">DISTRIBUTION OF NANOPARTICLES IN A POLYMER MATRIX PREDICTION</h1>'
st.markdown(title_main,unsafe_allow_html=True)
header1_main='<h2 style="text-align:center; font-weight: bolder;">Problem Description</h2>'
st.markdown(header1_main,unsafe_allow_html=True)
st.write("""    Polymer nanocomposites (PNC) offer a broad range of properties that are intricately 
         connected to the spatial distribution of nanoparticles (NPs) in polymer matrices. 
         Understanding and controlling the distribution of NPs in a polymer matrix is a significantly challenging task.
         We aim to address this challenge via machine learning. In this website, we use Decision Tree Regression to predict the distribution of nanoparticles in a polymer matrix.""")
image=Image.open("polymer_nanoparticle.jpg")
st.image(image,caption="nanoparticle in a polymer matrix, distribution diagram of nanoparticle")
st.image("https://editor.analyticsvidhya.com/uploads/210362021-07-18%20(2).png",caption="artificial neural network")
st.write("    In this problem, we have 6 inputs including: amplitude of interaction between polymer-nanoparticle, nanoparticle-nanoparticle, diameter of nanoparticle, phi, chain length of polymer and distance range")
st.write("    While output is function g(r)- distribution of nanoparticle.")
st.write("For more information, please read this article:  [nanoNET: machine learning platform for predicting nanoparticles distribution in a polymer matrix](https://pubs.rsc.org/en/content/articlelanding/2023/sm/d3sm00567d/unauth)")

# input explaination:
ls1="""<ul>
        <li>Interaction polymer-nanoparticle: amplitube</li>
        <li>Interaction nanoparticle-nanoparticle: amplitube</li>
        <li>Diameter of nanoparticle: size of nanoparticle (sperical, in nanometer)</li>
        <li>Phi: represented by mass of nanoparticle per total volume</li>
        <li>Length of polymer chain: in nanometer</li>
        <li>Distance: range should be small (less than length of polymer chain)</li>
    </ul>"""

st.markdown(
    '''
    <style>
    .streamlit-expanderHeader {
        background-color: lightblue;
        color: black; # Adjust this for expander header color
        border-radius: 20px;
        text-align:center;
    }
    .streamlit-expanderContent {
        background-color: white;
        color: black; # Expander content color
    }
    </style>
    ''',
    unsafe_allow_html=True
)

with st.expander("Input explaination"):
    st.markdown(ls1,unsafe_allow_html=True)

if st.sidebar.button("Predict!"):
    header2_main='<h2 style="text-align:center; font-weight: bolder;">User Input Parameters</h2>'
    st.markdown(header2_main,unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Interaction: polymer-NP", "{}".format(max(df['Po_NP'])))
    col2.metric("Interaction: NP-NP", "{}".format(max(df['NP_NP'])))
    col3.metric("Diameter of NP", "{}".format(max(df['D_aim'])))
    col4,col5=st.columns(2)
    col4.metric("Number of NP", "{}".format(max(df['Phi'])))
    col5.metric("Length of polymer chain", "{}".format(max(df['Chain length'])))
    st.metric("Distance range: ", "From {} nm to {} nm".format(min(df['distance']),max(df['distance'])))
    # Load the model
    model = joblib.load('/mount/src/app/model.pkl')
    predictions1=model.predict(df)
    res= result_programme(max(df['Po_NP']),max(df['NP_NP']),max(df['D_aim']),max(df['Phi']),max(df['Chain length']))
    data_res=pd.DataFrame(res)
    frame1=data_res.iloc[:,6][0]
    frame2=data_res.iloc[:,7][0]
    st.write('Minimum distance in calculation: ',min(frame1))
    st.write('Maximum distance in calculation: ',max(frame1))
    st.subheader('Prediction')
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots()
    ax.plot(frame1,frame2, color='green',label='Calculation')
    ax.plot(df['distance'],predictions1, color="lightblue",label='Prediction')
    ax.set_xlabel('Distance (nm)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.set_xlim(min(df['distance']), max(df['distance']))
    #ax.set_title('Nanoparticle distribution Prediction')

    # Display the plot in Streamlit
    st.pyplot(fig)

    #Allow data download
    download = df
    df = pd.DataFrame(download)
    csv = df.to_csv(index=False)
    fn =  str(max(df['Po_NP']))+' - ' +str(max(df['NP_NP']))+str(max(df['D_aim']))+str(max(df['Phi']))+str(max(df['Chain length']))+' - '+str(min(df['distance']))+'/'+str(max(df['distance'])) + '.csv'
    down_load_button= st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=fn,
        mime='text/csv',
        type="primary",
    )
    if down_load_button:
        st.success("Successful download!")
    
