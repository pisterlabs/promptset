# %%
import pandas as pd
import numpy as np
import streamlit as st
import openai
from pandasai import PandasAI

# %%
#read csv
# df = pd.read_csv('./Sangrur/result.csv')
# df.head()
# #show all columns
# pd.set_option('display.max_columns', None)
# df.head()
# df_a = pd.read_csv('./Sangrur/result2.csv')
# #show features
# df2tempa = df[['features__attributes__objectid', 'features__attributes__schcd',
#        'features__attributes__schname', 'features__attributes__schcat',
#        'features__attributes__school_cat', 'features__attributes__schtype',
#        'features__attributes__school_typ', 'features__attributes__schmgt',
#        'features__attributes__management', 'features__attributes__rururb',
#        'features__attributes__location', 'features__attributes__pincode',
#        'features__attributes__dtname', 'features__attributes__udise_stco',
#        'features__attributes__stname', 'features__attributes__vilname',
#        'features__attributes__longitude', 'features__attributes__latitude',
#        'features__attributes__stcode11', 'features__attributes__dtcode11',
#        'features__attributes__sdtcode11', 'features__attributes__sdtname',
#        'features__geometry__x', 'features__geometry__y']]
# df2tempb = df_a[['features__attributes__objectid', 'features__attributes__schcd',
#        'features__attributes__schname', 'features__attributes__schcat',
#        'features__attributes__school_cat', 'features__attributes__schtype',
#        'features__attributes__school_typ', 'features__attributes__schmgt',
#        'features__attributes__management', 'features__attributes__rururb',
#        'features__attributes__location', 'features__attributes__pincode',
#        'features__attributes__dtname', 'features__attributes__udise_stco',
#        'features__attributes__stname', 'features__attributes__vilname',
#        'features__attributes__longitude', 'features__attributes__latitude',
#        'features__attributes__stcode11', 'features__attributes__dtcode11',
#        'features__attributes__sdtcode11', 'features__attributes__sdtname',
#        'features__geometry__x', 'features__geometry__y']]

#read all csv and concat
def read_csv_and_concat(file_path):
    df = pd.read_csv(file_path)
    df2temp = df[['features__attributes__objectid', 'features__attributes__schcd',
       'features__attributes__schname', 'features__attributes__schcat',
       'features__attributes__school_cat', 'features__attributes__schtype',
       'features__attributes__school_typ', 'features__attributes__schmgt',
       'features__attributes__management', 'features__attributes__rururb',
       'features__attributes__location', 'features__attributes__pincode',
       'features__attributes__dtname', 'features__attributes__udise_stco',
       'features__attributes__stname', 'features__attributes__vilname',
       'features__attributes__longitude', 'features__attributes__latitude',
       'features__attributes__stcode11', 'features__attributes__dtcode11',
       'features__attributes__sdtcode11', 'features__attributes__sdtname',
       'features__geometry__x', 'features__geometry__y']]
    return df2temp

def read_block(district):
    district = district.upper()
    path = "./blocks/LUDHIANA.xlsx"
    df = pd.read_excel(path,sheet_name='Database', header=1)
    #make sure to include all columns
    #filter district
    # st.write(df)
    df = df[df['DISTRICT_NAME'] == district]
    return df
    

# df2tempa = read_csv_and_concat('./Sangrur/result.csv')
# df2tempb = read_csv_and_concat('./Sangrur/result2.csv')
# df2tempc = read_csv_and_concat('./Sangrur/result3.csv')

#read all files from Sangrur folder and read csv and concat and add to a list
dfs = []
#read all files from Sangrur folder and read csv and concat and add to a list
import os
for file in os.listdir('./Sangrur'):
    if file.endswith('.csv'):
        dfs.append(read_csv_and_concat('./Sangrur/' + file))

df2temp = pd.concat(dfs).drop_duplicates()
#state punjab
df2temp = df2temp[df2temp['features__attributes__stname'] == 'PUNJAB']

# df2temp = pd.concat([df2tempa, df2tempb]).drop_duplicates()
#merge both df without 
# %%
# filter out district with name sangrur
# df2 = df2[df2['features__attributes__dtname'] == 'Sangrur']
# df2['LAT'] = df2['features__attributes__latitude']
# df2['LON'] = df2['features__attributes__longitude']
# df2

# %%
def filter_district(df, district):
    df = df[df['features__attributes__dtname'] == district]
    df['LAT'] = df['features__attributes__latitude']
    df['LON'] = df['features__attributes__longitude']
    return df

def filter_school_type(df, school_type):
    df = df[df['features__attributes__school_typ'] == school_type]
    df['LAT'] = df['features__attributes__latitude']
    df['LON'] = df['features__attributes__longitude']
    return df

# %%
# #use streamlit to show data
# st.title('Schools in Sangrur')
# st.write(df2)
# #show map of sangrur
# st.map(df2)
# map_type = st.selectbox('Select map type', ['stamen', 'carto', 'openstreetmap', 'esri', 'stamenterrain', 'stamentoner', 'stamenwatercolor', 'stamenterrain'])
# st.map(df2[['LAT', 'LON']], zoom=10, use_container_width=True, height=500, tooltip=df2['features__attributes__schname'], map_type=map_type)
# #show map of sangrur with markers and zoom

#show other details of the school when clicked on marker
#use folium to show map
import folium
from streamlit_folium import folium_static
#put bcg logo on top of the site
from PIL import Image
image = Image.open('./logo.png')

#create columns

#add a pill with text 
# st.markdown("<h4 style='text-align: left; color: black;'>School GIS Mapping</h4>", unsafe_allow_html=True)
#add bcg logo on top of the sidebar
image = Image.open('./logo.png')
st.image(image, width=100)
#add a footer with texts
st.sidebar.markdown("<h4 style='text-align: left; color: black;'>Developed by: BCG Social Impact</h4>", unsafe_allow_html=True)


def school_maps():
#add columns
    col1, col2 = st.columns([1, 1])

    district = col1.selectbox('Select District', df2temp['features__attributes__dtname'].unique())
    df2 = filter_district(df2temp, district)

    #select school type and have all as an option
    school_type = col2.selectbox('Select School Type', df2['features__attributes__school_typ'].unique())
    df2 = filter_school_type(df2, school_type)

    col1, col2 = st.columns([1, 1])
    #add a button to reset the filters
    if col1.button('Reset Filters'):
        df2 = filter_district(df2temp, district)
        # df2['LAT'] = df2['features__attributes__latitude']
        # df2['LON'] = df2['features__attributes__longitude']

    #add a button to show block wise data of the district
    if col2.button('Show Block Wise Data'):
        df_block = read_block(district)
        df_block.fillna(0, inplace=True)
        #merge with key as school name remove everything else
        df2= df_block.merge(df2, left_on='School_Name', right_on='features__attributes__schname', how='left')
        #if none in lat and lon them remove column
        #first count how many lat or lon are none
        st.write("Schools not scraped: {}".format(df2['LAT'].isna().sum()))
        df2.dropna(subset=['LAT', 'LON'], inplace=True)
        #filter by nsqf and vocational or both
        

        
        #st.write(df2)
        #convert all none to 0

    # st.write(df2)7

    m = folium.Map(location=[df2['LAT'].mean(), df2['LON'].mean()], zoom_start=10)
    #let the map cover the whole screen
    folium.TileLayer('cartodbpositron').add_to(m)
  


    for i in range(0,len(df2)):
        mypopup = "School Name:{} \n Type:{} \n Management:{} \n Location:{} \n Pincode:{}".format(df2.iloc[i]['features__attributes__schname'], df2.iloc[i]['features__attributes__school_typ'], df2.iloc[i]['features__attributes__management'], df2.iloc[i]['features__attributes__sdtname'], df2.iloc[i]['features__attributes__pincode'])
        html = '''
        <h3>School Name:{}</h3>
        <p>Type:{}</p>
        <p>Management:{}</p>
        <p>Location:{}</p>
        <p>Pincode:{}</p>
        '''.format(df2.iloc[i]['features__attributes__schname'], df2.iloc[i]['features__attributes__school_typ'], df2.iloc[i]['features__attributes__management'], df2.iloc[i]['features__attributes__sdtname'], df2.iloc[i]['features__attributes__pincode'])
        iframe = folium.IFrame(html=html, width=200, height=200)
        if 'NSQF?' in df2.columns:
            html = '''
            <h3>School Name:{}</h3>
            <p>Type:{}</p>
            <p>Management:{}</p>
            <p>Location:{}</p>
            <p>Pincode:{}</p>
            <p>NSQF:{}</p>
            <p>Vocational:{}</p>
            '''.format(df2.iloc[i]['features__attributes__schname'], df2.iloc[i]['features__attributes__school_typ'], df2.iloc[i]['features__attributes__management'], df2.iloc[i]['Block Name'], df2.iloc[i]['features__attributes__pincode'], df2.iloc[i]['NSQF?'], df2.iloc[i]['Vocational'])
            iframe = folium.IFrame(html=html, width=200, height=200)
            #add markers with different colors for nsqf and vocational
            if df2.iloc[i]['NSQF?'] == 'yes' and df2.iloc[i]['Vocational'] == 'no':
                folium.Marker(
                    location=[df2.iloc[i]['LAT'], df2.iloc[i]['LON']],
                    popup=folium.Popup(iframe, max_width=200),
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
            elif df2.iloc[i]['Vocational'] == 'yes' and df2.iloc[i]['NSQF?'] == 'no':
                folium.Marker(
                    location=[df2.iloc[i]['LAT'], df2.iloc[i]['LON']],
                    popup=folium.Popup(iframe, max_width=200),
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
            elif df2.iloc[i]['Vocational'] == 'yes' and df2.iloc[i]['NSQF?'] == 'yes':
                folium.Marker(
                    location=[df2.iloc[i]['LAT'], df2.iloc[i]['LON']],
                    popup=folium.Popup(iframe, max_width=200),
                    icon=folium.Icon(color='black', icon='info-sign')
                ).add_to(m)
        else:
            folium.Marker(
                location=[df2.iloc[i]['LAT'], df2.iloc[i]['LON']],
                popup=folium.Popup(iframe, max_width=200),
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(m)
    folium_static(m)

    #show legend outside map show color and what it means
    if 'NSQF?' in df2.columns:
        st.write("Red: NSQF, Blue: Vocational, Black: Both, Green: Neither")





import matplotlib.pyplot as plt
def pandas_AI():
    #get gpt api key
    gpt_key = st.secrets['gpt_key']
    # Sample DataFrame
    path = './blocks/LUDHIANA.xlsx'
    df = pd.read_excel(path,sheet_name='Database', header=1)
    # Instantiate a LLM
    from pandasai.llm.openai import OpenAI
    llm = OpenAI(api_token=gpt_key)
    pandas_ai = PandasAI(llm)
    #welcome message for pandasai and how to use it
    st.write("Welcome to AnalysisAI. It can answer questions about your data, generate plots, and more. To use it, simply type a question about your data in the box below and press enter.")
    st.write(df)
    #get the text from the user
    text = st.text_input('Enter your question')
    #get the answer from pandasai
    answer = pandas_ai.run(df, text)

    #answer can be a pandas dataframe or plot
    st.write(answer)


    #get the text from the user


page_to_show = {
    "School Maps": school_maps,
    "Analysis AI": pandas_AI
}
name_page = st.sidebar.selectbox("Go to", page_to_show.keys())
page_to_show[name_page]()




# %%
#run streamlit
#streamlit run Mapping.py



