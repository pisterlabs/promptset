import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import AgglomerativeClustering as AC
import openai


plt.style.use('seaborn')

openai.api_key = st.secrets['APIKEY']

rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}

slot2num = {'Midnight':0,
                'Morning':1,
                'Afternoon':2,
                'Night':3}

def scale_df(df,axis=0):
    '''
    A utility function to scale numerical values (z-scale) to have a mean of zero
    and a unit variance.
    '''
    return (df - df.mean(axis=axis)) / df.std(axis=axis)

def plot_hmap(df, ix=None, cmap='Reds'):
    '''
    A function to plot heatmaps that show temporal patterns
    '''
    if ix is None:
        ix = np.arange(df.shape[0])
    fig, ax = plt.subplots()
    plt.rcParams.update(rc)
    plt.imshow(df.iloc[ix,:], cmap=cmap)
    plt.colorbar(fraction=0.03)
    plt.yticks(np.arange(df.shape[0]), df.index[ix])
    plt.xticks(np.arange(df.shape[1]), list(df.columns))
    plt.grid(False)
    st.pyplot(fig)
    
def scale_and_plot(df, ix = None):
    '''
    A wrapper function to calculate the scaled values within each row of df and plot_hmap
    '''
    df_marginal_scaled = scale_df(df.T).T
    if ix is None:
        ix = AC(4).fit(df_marginal_scaled).labels_.argsort() # a trick to make better heatmaps
    cap = np.min([np.max(df_marginal_scaled.values), np.abs(np.min(df_marginal_scaled.values))])
    df_marginal_scaled = np.clip(df_marginal_scaled, -1*cap, cap)
    plot_hmap(df_marginal_scaled, ix=ix)
    
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


df = pd.read_csv('data/Chicago_crimes.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df.Date = pd.to_datetime(df.Date)
df.index = pd.DatetimeIndex(df.Date)

def app():
    st.title('Shadows in the Sun - The guilty secret in a busy city')

    # Image
    image = Image.open('skyline.jpg')
    st.image(image, caption='Hell is empty. The devil is on Chicago.')

    # Artificial Intelligence Dialogue
    st.write('')
    st.write('')
    st.subheader('Artificial Intelligence Dialogue')
    user_inputs = st.text_input('Your question: ')
    st.write('')
    st.write('')

    query = user_inputs.strip()

    if query:
        template =  "Human: What are your impressions of Chicago?\n"+ \
                    "AI: I think Chicago is the capital of sin\n"
        inputs = f'{template}Human: {query}\nAI:'

        response = openai.Completion.create(
        model="text-davinci-002",
        prompt=inputs,
        temperature=0.9,
        max_tokens=150,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
        )
        ai_output = response['choices'][0]['text'].strip()
        st.text_area('Your Security Advisor: ', ai_output)

    # 是否通过案件类型查看案件随时间的走势
    st.text('')
    st.text('')
    row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
    with row1_1:
        st.text('')
        st.text('')
        select_box = df['Primary Type'].unique().tolist()
        select_box.append('ALL TYPES')
        type_filter = st.selectbox(
            'Choosing a Crime type',
            select_box)
        days_filter = st.slider('Dats of rolling sum', 1, 365, 30)
        st.markdown('Rolling day sum is the sum of the first n days, and as the number of days summed gradually increases, the curve shows a clear downward trend, indicating that policing in Chicago has improved between 2012 and 2016.')
    with row1_2:
        st.subheader(f'Rolling sum of {type_filter.lower()} from 2012 - 2016 by {days_filter} days')
        plt.rcParams.update(rc)
        fig, ax = plt.subplots()
        if type_filter != 'ALL TYPES':
            type_df = df[df['Primary Type']==type_filter]
            type_df.resample('D').size().rolling(days_filter).sum().plot(color='#cf0c0c')
        else:
            df.resample('D').size().rolling(days_filter).sum().plot(color='#cf0c0c')
        ax.set_xlabel('')
        st.pyplot(fig)



    # 根据时间段查看各案件的分布
    st.text('')
    st.text('')
    row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
    with row2_1:
        slots = ['Midnight', 'Morning', 'Afternoon', 'Night']
        st.text('')
        st.text('')
        slot_filter = st.multiselect(
            'Choosing the slots',
            slots,
            slots)
        ascending_filter = st.checkbox(
            'In ascending order'
            )
        arrest_filter = st.checkbox(
            'Only see NOT arrested'
            )
        domestic_filter = st.checkbox(
            'Only see domestic violence'
            )
        st.markdown('The day is divided into four equal time periods: midnight from 00:00 to 6:00, morning from 6:00 to 12:00, afternoon from 12:00 to 18:00 and night from 18:00 to 24:00.')

    with row2_2:
        st.subheader('Distribution of cases in different time periods')
        if slot_filter:
            slot_filter_list = [slot2num[i] for i in slot_filter]
            slot_df = df[df.Slot.isin(slot_filter_list)]

            if arrest_filter:
                slot_df = slot_df[slot_df['Arrest'] == False]
            if domestic_filter:
                slot_df = slot_df[slot_df['Domestic'] == True]

            plt.rcParams.update(rc)
            fig, ax = plt.subplots()
            if ascending_filter:
                slot_df.groupby([slot_df['Primary Type']]).size().sort_values(ascending=True).plot(kind='barh', color='#cf0c0c')
            else:
                slot_df.groupby([slot_df['Primary Type']]).size().plot(kind='barh', color='#cf0c0c')
            ax.set_ylabel('')
            ax.set_xlim(0, 50000)
            st.pyplot(fig)

        else:
            st.warning('Select at least one slot')


    # 案件发生的时间段与案件类型和案发地点的关系
    st.text('')
    st.text('')
    row3_spacer1, row3_1, row3_spacer2, row3_2, row3_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
    with row3_1:
        st.text('')
        st.text('')
        type_or_location = st.radio(
            'Select the category',
            ('type', 'location'))
        st.markdown('The figure on the right shows the distribution of case types or crime locations in a day, the darker the color represents the higher the frequency, according to the heat map, you can clearly see the frequency change of different case types or crime locations with time.')
    with row3_2:
        st.subheader('Temporal analysis of crime rates by type and location')
        button2column = {'type': 'Primary Type',
                    'location': 'Location Description'}
        slot_by_select = df.pivot_table(values='ID', index=button2column[type_or_location], columns=df.index.hour, aggfunc=np.size).fillna(0)
        scale_and_plot(slot_by_select)