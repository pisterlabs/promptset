import os
import re
import json
import warnings
import base64
import hashlib
import openai
import streamlit as st

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from itertools import combinations

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, classification_report, roc_curve, auc, silhouette_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from scipy.stats import zscore

from wordcloud import WordCloud

# from autoviz.AutoVizClass import AutoVizClass
import dtale
import streamlit.components.v1 as components

from lazypredict.Supervised import LazyClassifier

from utils import get_answer_csv
from load_dataframe import LoadDataframe
from data_viz import DataViz
from dataviz.barchart import BarChart
from dataviz.piechart import PieChart


# Fungsi untuk mengenkripsi password
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


# Fungsi untuk memeriksa password yang di-hash
def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


def login():
    # CSS untuk menyesuaikan posisi "Ulikdata", gaya tombol, dan gaya kolom
    st.markdown("""
        <style>
            #ulikdata {
                position: absolute;
                top: 10px;
                left: 10px;
                font-weight: bold;
                font-style: italic;
                z-index: 999;
            }
            .stButton>button {
                background-color: lightblue;
                color: white;
            }
            .stButton:hover>button {
                background-color: pink;
            }
            .login-bg {
                background-color: #6a5acd;  /* Biru keunguan */
                color: white;  /* Warna font putih */
            }
        </style>
    """,
                unsafe_allow_html=True)

    # Tulisan "Ulikdata" dengan ID yang kita definisikan di CSS
    st.markdown('<div id="ulikdata">Ulikdata</div>', unsafe_allow_html=True)
    st.markdown("---")

    _, col2, _ = st.columns(3)

    # Tambahkan gambar dari Google Drive di kolom pertama (col1)
    # col1.image(
    #     'https://drive.google.com/uc?export=view&id=1Kzbvjj3M_HxcvPv-JJZaZUvvwi2ltGKL',
    #     use_column_width=True)
    # st.image('https://drive.google.com/uc?export=view&id=1Kzbvjj3M_HxcvPv-JJZaZUvvwi2ltGKL', use_column_width=True)
    # https://drive.google.com/file/d/1Kzbvjj3M_HxcvPv-JJZaZUvvwi2ltGKL/view?usp=sharing

    # Gunakan kelas CSS di dalam kolom kedua untuk background dan warna font
    with col2:
        st.markdown('<div class="login-bg">', unsafe_allow_html=True)

        st.subheader("Enter your email that is registered on ulikdata.com")
        usermail = st.text_input("email")
        # password = st.text_input("Password", type='password')

        # st.write("If you haven't registered, please register for free at ulikdata.com")
        st.markdown(
            "[If you haven't registered, please register here!](https://www.ulikdata.com)"
        )

        # hashed_pswd = make_hashes("test")

        # # Establish a connection to the database
        # connection = psycopg2.connect(
        #     host="172.104.32.208",
        #     port="3768",
        #     database="public",
        #     user=st.secrets['user_db'],
        #     password=st.secrets['pass_db']
        # )

        # # Membuat kursor
        # cursor = connection.cursor()

        # # Menjalankan query SQL untuk mengecek keberadaan email
        # cursor.execute("SELECT EXISTS(SELECT 1 FROM account WHERE email = %s)", (usermail,))

        # # Mengambil hasil query
        # exists = cursor.fetchone()[0]

        # # Menutup kursor dan koneksi
        # cursor.close()
        # connection.close()

        # if exists == True:
        if usermail == "founder_superuser@gmail.com":
            st.session_state.logged = True
            st.experimental_rerun()
        else:
            st.write("""Your email has not been registered.""")

        # if check_hashes(password, hashed_pswd):
        #     st.session_state.logged = True
        #     st.experimental_rerun()

        if st.button("Login", key='login_button'):
            if 'button_clicked' in st.session_state and st.session_state.logged:
                st.session_state.logged = True
                st.experimental_rerun()
                # if check_hashes(password, hashed_pswd):
                #     st.session_state.logged = True
                #     st.experimental_rerun()
            else:
                st.write("""Your email has not been registered.""")
                # st.session_state.button_clicked = True
                # st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)


def display(obj, *args, **kwargs):
    """Mock the Jupyter display function to use show() instead."""
    try:
        obj.show()
    except AttributeError:
        st.write("Object does not have a show() method.")


import sys

sys.modules["__main__"].display = display

hide_menu = """
<style>

.stActionButton {
  visibility: hidden !important;
}
.css-10pw50 {
  visibility: hidden !important;
.css-10pw50 {
    display: none !important;
}
</style>
"""

st.set_page_config(page_title="Ulikdata", page_icon=":tada:", layout="wide")


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")

## or use this:
## CSS untuk tombol
#  .stButton>button {
#       background-color: lightblue;
#       color: white;
# }
#  .stButton:hover>button {
#       background-color: pink;
# }
# .css-ztfqz8 {
#     display: none !important;
# }


def request_prompt(input_pengguna,
                   df,
                   schema_str,
                   rows_str,
                   style,
                   error_message=None,
                   previous_script=None,
                   retry_count=0):
    '''
    versi code + penjelasan prompt
    # messages = [
    #     {"role": "system", "content": "I only response with python syntax streamlit version, no other text explanation."},
    #     {"role": "user", "content": f"""I have a dataframe name df with the following column schema: {schema_str}, and 2 sample rows: {rows_str}.
    #     1. Respon pertanyaan atau pernyataan ini: {input_pengguna}.
    #     2. My dataframe already load previously, named df, use it, do not reload the dataframe.
    #     3. Only respond with python scripts in streamlit version without any text.
    #     4. Start your response with “import”.
    #     5. Show all your response to streamlit apps.
    #     6. Use Try and Except.
    #     7. Pay attention to the column type before creating the script.
    '''

    if style == 'General Question':
        script = get_answer_csv(df, input_pengguna)

        # messages = [{
        #     "role":
        #     "system",
        #     "content":
        #     "I only response with python syntax streamlit version, no other text explanation."
        # }, {
        #     "role":
        #     "user",
        #     "content":
        #     f"""I have a dataframe name df with the following column schema: {schema_str}, and 2 sample rows: {rows_str}.
        #     1. {input_pengguna}.
        #     2. My dataframe already load previously, named df, use it, do not reload the dataframe.
        #     3. Respond with scripts without any text.
        #     4. Respond in plain text code.
        #     5. Don’t start your response with “Sure, here are”.
        #     6. Start your response with “import”.
        #     7. Don’t give me any explanation about the script. Response only with python code in a plain text.
        #     8. Do not reload the dataframe.
        #     9. Use Try and Except for each syntax, Except with pass.
        #     10. Tulis code untuk menjawab poin 1 dalam versi streamlit untuk dieksekusi. Sesuaikan code-nya dengan data types dari {schema_str}.
        #     11. Optimalkan script-nya agar tidak panjang."""
        # }]
        # response = openai.ChatCompletion.create(
        #     # model="gpt-3.5-turbo-16k",
        #     model="gpt-3.5-turbo",
        #     # model="gpt-4",
        #     messages=messages,
        #     max_tokens=3000,
        #     temperature=0)
        # script = response.choices[0].message['content']
    else:
        # versi 1 prompt
        messages = [{
            "role":
            "system",
            "content":
            "I only response with python syntax streamlit version, no other text explanation."
        }, {
            "role":
            "user",
            "content":
            f"""I have a dataframe name df with the following column schema: {schema_str}, and 2 sample rows: {rows_str}. 
            1. {input_pengguna}. 
            2. My dataframe already load previously, named df, use it, do not reload the dataframe.
            3. Respond with scripts without any text. 
            4. Respond in plain text code. 
            5. Don’t start your response with “Sure, here are”. 
            6. Start your response with “import”.
            7. Don’t give me any explanation about the script. Response only with python code in a plain text.
            8. Do not reload the dataframe.
            9. Use Try and Except for each syntax, Except with pass.
            10. Give a Title inside the chart for each visualization. Don't use st.title or st.subheader.
            11. Use unique streamlit widgets.
            12. Use Plotly library for visualization.
            13. Pay attention to the dataframe schema, don't do any convert."""
        }]
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo-16k",
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=messages,
            max_tokens=3000,
            temperature=0)
        script = response.choices[0].message['content']

    # if error_message and previous_script:
    #     messages.append({
    #         "role":
    #         "user",
    #         "content":
    #         f"Solve this error: {error_message} in previous Script : {previous_script} to "
    #     })

    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo-16k",
    #     # model="gpt-3.5-turbo",
    #     # model="gpt-4",
    #     messages=messages,
    #     max_tokens=3000,
    #     temperature=0)
    # script = response.choices[0].message['content']

    return script


# Jangan diubah yg ini
def request_story_prompt(schema_str,
                         rows_str,
                         min_viz,
                         api_model,
                         style='Plotly'):
    # messages = [
    #     {"role": "system", "content": "Aku akan membuat laporan untukmu."},
    #     {"role": "user", "content": f"""Buatkan laporan berbentuk insights yang interpretatif dari data berikut:  {dict_stats}.
    #     Jika ada pesan error, skip saja tidak usah dijelaskan. Tidak usah dijelaskan bahwa kamu membaca dari dictionary.
    #     Tulis dalam 3000 kata. Tambahkan kesimpulan dan insights yang aplikatif dalam bisnis. Jelaskan dalam bentuk poin-poin."""}
    # ]

    # Versi penjelasan dan code
    messages = [{
        "role":
        "system",
        "content":
        f"I will create a long article for you in the form of analysis and visualization in {style} scripts to be displayed in Streamlit. Every script should start with 'BEGIN_CODE' and end with 'END_CODE'."
    }, {
        "role":
        "user",
        "content":
        f"""Create an article in the form of insights that are insightful from data with the schema: {schema_str}, and the first 2 sample rows as an illustration: {rows_str}.
        My dataframe has been loaded previously, named 'df'. Use it directly; do not reload the dataframe, and do not redefine the dataframe.
        The article should start with an introductory paragraph in plain text, followed by introduction for the first insight and the visualization of the first insight with {style} library for visualization and show in streamlit.
        Then continue with an introduction paragraph in for insight 2, followed by the visualization of the second with {style} library for visualization and show in streamlit.
        And so on, up to a minimum of {min_viz} insights, do not provide under {min_viz} number of insights, the minimum is {min_viz}.
        Display in order: introductory, introduction for insight 1, visualization for insight 1, introduction for insight 2, visualization for insight 2, and so on.
        Every script should start with 'BEGIN_CODE' and end with 'END_CODE'.
        Use df directly; it's been loaded before, do not reload the df, and do not redefine the df.
        Visualize with {style} in streamlit, different columns for different insight, the insights must be unique and interesting, and it's {min_viz} most important insights from the data.
        
        """
    }]
    # Optimize the script for efficiency and minimize the number of lines.

    # messages = [{
    #     "role":
    #     "system",
    #     "content":
    #     f"I will create a long article for you in the form of analysis and visualization in {style} scripts to be displayed in Streamlit. Every script should start with 'BEGIN_CODE' and end with 'END_CODE'."
    # }, {
    #     "role":
    #     "user",
    #     "content":
    #     f"""Create an article in the form of insights that are insightful from data with the schema: {schema_str}, and the first 2 sample rows as an illustration: {rows_str}.
    #     My dataframe has been loaded previously, named 'df'. Use it directly; do not reload the dataframe, and do not redefine the dataframe.
    #     The article should start with an introductory paragraph in plain text, followed by introduction for the first chart type and the visualization of the first chart type with {style} library for visualization and show in streamlit.
    #     Then continue with an introduction paragraph in for chart type 2, followed by the visualization of the second with {style} library for visualization and show in streamlit.
    #     And so on, up to a minimum of {min_viz} chart type, do not provide under {min_viz} number of chart type, the minimum is {min_viz}.
    #     Display in order: introductory, introduction for chart type 1, visualization for chart type 1, introduction for chart type 2, visualization for chart type 2, and so on.
    #     Every script should start with 'BEGIN_CODE' and end with 'END_CODE'.
    #     Use df directly; it's been loaded before, do not reload the df, and do not redefine the df.
    #     Give a clear {style} title.
    #     Optimize the script for efficiency and minimize the number of lines.
    #     Use looping for columns if multiple visualizations of the same type can run simultaneously. For example chart type 1 is distribution, chart type 2 is correlation, chart type 3 is wordcloud, and soon.
    #     Pay attention to the dataframe schema for best and interesting chart types."""
    # }]
    # Every script should start with 'BEGIN_CODE' and end with 'END_CODE'.
    #
    if api_model == 'GPT3.5':
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            # model="gpt-4",
            messages=messages,
            max_tokens=3000,
            temperature=0)
        script = response.choices[0].message['content']
    else:
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            messages=messages,
            max_tokens=3000,
            temperature=0)
        script = response.choices[0].message['content']

    return script


# Function to display a histogram
def show_histogram(df):
    left_column, right_column = st.columns(2)
    # Selecting the numeric column
    column = left_column.selectbox(
        'Select a Numeric Column for Histogram:',
        df.select_dtypes(include=['number']).columns.tolist())

    # Customization options
    bins = right_column.slider('Select Number of Bins:', 5, 50,
                               15)  # Default is 15 bins
    kde = left_column.checkbox('Include Kernel Density Estimate (KDE)?',
                               value=True)  # Default is to include KDE
    color = right_column.color_picker('Pick a color for the bars:',
                                      '#3498db')  # Default is a shade of blue

    # Plotting the histogram using Seaborn
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], bins=bins, kde=kde, color=color)

    # Rendering the plot in Streamlit
    st.pyplot(plt)


# Function to display a box plot
def show_box_plot(df):
    st.subheader("Box Plot")
    left_column, middle_column, right_column = st.columns(3)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    y_column = left_column.selectbox('Select a Numeric Column for Y-axis:',
                                     numeric_columns)
    x_column = middle_column.selectbox(
        'Select a Categorical Column for X-axis (Optional):',
        [None] + categorical_columns)

    show_swarm = right_column.checkbox('Show Swarm Plot?')

    if x_column:
        cat_palette_option = st.selectbox(
            'Choose a color palette for Categorical Box Plot:',
            sns.palettes.SEABORN_PALETTES)
        cat_palette = sns.color_palette(cat_palette_option,
                                        len(df[x_column].unique()))
    else:
        color_option = st.color_picker('Pick a Color for Box Plot',
                                       '#add8e6')  # Default light blue color

    fig, ax = plt.subplots(figsize=(10, 6))

    if x_column:
        sns.boxplot(x=x_column,
                    y=y_column,
                    data=df,
                    ax=ax,
                    palette=cat_palette)
    else:
        sns.boxplot(x=x_column, y=y_column, data=df, ax=ax, color=color_option)

    if show_swarm:
        sns.swarmplot(x=x_column,
                      y=y_column,
                      data=df,
                      ax=ax,
                      color='black',
                      size=3)

    sns.despine(left=True, bottom=True)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.title('Box Plot of ' + y_column, fontsize=16, fontweight="bold")
    plt.ylabel(y_column, fontsize=12)
    plt.xlabel(x_column if x_column else '', fontsize=12)

    if x_column:
        categories = df[x_column].unique()
        for category in categories:
            subset = df[df[x_column] == category][y_column]
            median = subset.median()
            plt.annotate(f'Median: {median}',
                         xy=(categories.tolist().index(category), median),
                         xytext=(-20, 20),
                         textcoords='offset points',
                         arrowprops=dict(arrowstyle='->'),
                         fontsize=10)
    else:
        median = df[y_column].median()
        plt.annotate(f'Median: {median}',
                     xy=(0, median),
                     xytext=(-20, 20),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->'),
                     fontsize=10)

    st.pyplot(fig)


# Function to display scatter plot
def show_scatter_plot(df):
    st.subheader("Scatter Plot")
    left_column, middle_column, right_column = st.columns(3)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    col1 = left_column.selectbox('Select the first Numeric Column:',
                                 numeric_columns,
                                 index=0)
    col2 = middle_column.selectbox('Select the second Numeric Column:',
                                   numeric_columns,
                                   index=1)
    hue_col = right_column.selectbox(
        'Select a Categorical Column for Coloring (Optional):',
        [None] + categorical_columns)
    size_option = st.slider('Select Point Size:',
                            min_value=1,
                            max_value=10,
                            value=5)
    show_regression_line = left_column.checkbox('Show Regression Line?')
    annotate_points = middle_column.checkbox('Annotate Points?')

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(x=col1,
                    y=col2,
                    hue=hue_col,
                    data=df,
                    s=size_option * 10,
                    ax=ax,
                    palette='viridis' if hue_col else None)

    if show_regression_line:
        sns.regplot(x=col1,
                    y=col2,
                    data=df,
                    scatter=False,
                    ax=ax,
                    line_kws={'color': 'red'})

    if annotate_points:
        for i, txt in enumerate(df.index):
            ax.annotate(txt, (df[col1].iloc[i], df[col2].iloc[i]), fontsize=8)

    sns.despine(left=True, bottom=True)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    plt.title(f'Scatter Plot of {col1} vs {col2}',
              fontsize=16,
              fontweight="bold")
    plt.xlabel(col1, fontsize=12)
    plt.ylabel(col2, fontsize=12)

    st.pyplot(fig)


# Function to display correlation matrix
def show_correlation_matrix(df):
    st.subheader("Correlation Matrix")
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=['number'])

    # Compute the correlation matrix
    corr = numerical_df.corr()

    # Palette options
    palette_options = [
        'coolwarm', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    ]
    selected_palette = st.selectbox('Select a color palette:', palette_options)

    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(corr, annot=False,
                     cmap=selected_palette)  # Set annot=False temporarily

    # Get tick label size from the x-axis (you can also get it from y-axis if preferred)
    tick_label_size = plt.xticks()[1][0].get_size()

    # Create annotations with the desired font size
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(i + 0.5,
                     j + 0.5,
                     format(corr.iloc[i, j], '.2f'),
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=tick_label_size)

    st.pyplot(plt)

    # Button to download the correlation table as CSV
    if st.button('Download correlation table as CSV'):

        # Convert DataFrame to CSV
        csv_file = corr.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()  # Bytes to string

        # Create download link
        href = f'<a href="data:file/csv;base64,{b64}" download="correlation_table.csv">Click here to download the CSV file</a>'
        st.markdown(href, unsafe_allow_html=True)


# Function to perform PCA
def perform_pca(df):
    st.subheader("Pricipal COmponent Analysis (PCA)")
    numeric_df = df.select_dtypes(include=['number'])
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # User options
    n_components = st.slider('Select Number of Principal Components:', 1,
                             min(numeric_df.shape[1], 10), 2)
    scaling_option = st.selectbox(
        'Select Scaling Option:',
        ['StandardScaler', 'MinMaxScaler', 'RobustScaler'])
    color_by = st.selectbox('Color By Categorical Column:',
                            [None] + categorical_columns)
    show_scree_plot = st.checkbox('Show Scree Plot', value=False)

    # Scaling
    if scaling_option == 'StandardScaler':
        scaler = StandardScaler()
    elif scaling_option == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaling_option == 'RobustScaler':
        scaler = RobustScaler()

    scaled_data = scaler.fit_transform(numeric_df)

    # PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    # Membuat DataFrame dengan hasil PCA dan data aktual
    pca_df = pd.DataFrame(data=pca_result,
                          columns=[
                              f'Principal Component {i}'
                              for i in range(1, n_components + 1)
                          ])
    combined_df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)

    # Tombol Unduh
    csv_file = combined_df.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()  # Beberapa byte dance
    href = f'<a href="data:file/csv;base64,{b64}" download="pca_result.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)

    # Scatter plot
    fig, ax = plt.subplots()
    if color_by:
        sns.scatterplot(x=pca_result[:, 0],
                        y=pca_result[:, 1],
                        hue=df[color_by],
                        palette='viridis')
    else:
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    st.pyplot(fig)

    # Scree plot
    if show_scree_plot:
        fig, ax = plt.subplots()
        plt.bar(range(1,
                      len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance')
        st.pyplot(fig)

    # Loadings
    loadings = pd.DataFrame(pca.components_.T,
                            columns=[f'PC{i+1}' for i in range(n_components)],
                            index=numeric_df.columns)
    st.write("Loadings:")
    st.write(loadings)


# Function to show outliers using Z-score
def show_outliers(df):
    st.subheader("Outliers Detection")
    column = st.selectbox(
        'Select a Numeric Column for Outlier Detection:',
        df.select_dtypes(include=['number']).columns.tolist())
    values = df[column].dropna()
    z_scores = np.abs(stats.zscore(values))
    threshold = 2
    outliers = np.where(z_scores > threshold)

    st.write(f"Outliers found at index positions: {outliers}")

    # Plotting the data
    plt.figure(figsize=(14, 6))
    plt.scatter(range(len(values)), values, label='Data')
    plt.axhline(y=np.mean(values) + threshold * np.std(values),
                color='r',
                linestyle='--',
                label='Upper bound')
    plt.axhline(y=np.mean(values) - threshold * np.std(values),
                color='r',
                linestyle='--',
                label='Lower bound')

    # Annotating the outliers
    for idx in outliers[0]:
        plt.annotate(f'Outlier\n{values.iloc[idx]}', (idx, values.iloc[idx]),
                     textcoords="offset points",
                     xytext=(-5, 5),
                     ha='center',
                     arrowprops=dict(facecolor='red',
                                     arrowstyle='wedge,tail_width=0.7',
                                     alpha=0.5))

    plt.title(f'Outlier Detection in {column}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    st.pyplot(plt)


# Function to perform Shapiro-Wilk normality test
def perform_shapiro_wilk_test(df):
    st.subheader("Normality Test")
    column = st.selectbox(
        'Select a Numeric Column for Normality Testing:',
        df.select_dtypes(include=['number']).columns.tolist())
    data = df[column].dropna()
    _, p_value = stats.shapiro(data)
    if p_value > 0.05:
        st.write(
            f"The data in the column '{column}' appears to be normally distributed (p-value = {p_value})."
        )
    else:
        st.write(
            f"The data in the column '{column}' does not appear to be normally distributed (p-value = {p_value})."
        )

    # Plotting the histogram
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data, bins=15, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Plotting the Q-Q plot
    plt.subplot(1, 2, 2)
    sm.qqplot(data, line='s', ax=plt.gca())
    plt.title(f'Q-Q Plot of {column}')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')

    st.pyplot(plt)


# Function to perform Linear Regression
def perform_linear_regression(df):
    st.subheader("Linear Regression")
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    X_columns = st.multiselect('Select Feature Columns:',
                               numeric_columns,
                               default=[numeric_columns[0]])
    if not X_columns:  # If no features are selected
        st.warning('Please select feature columns.')
        return

    y_column = st.selectbox(
        'Select Target Column:',
        df.select_dtypes(include=['number']).columns.tolist())
    test_size = st.slider('Select Test Size for Train-Test Split:', 0.1, 0.5,
                          0.2)
    fit_intercept = st.checkbox('Fit Intercept?', value=True)

    X = df[X_columns]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=42)
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("Model Coefficients:", model.coef_)
    st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))

    # Scatter plot of predicted vs actual values
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(y_test), max(y_test)],
             [min(y_test), max(y_test)],
             color='red')  # Identity line
    st.pyplot()
    plt.clf()

    # Residual plot
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='--')
    st.pyplot()


# Function to perform Logistic Regression
def perform_logistic_regression(df):
    st.subheader("Logistic Regression")

    left_column, right_column = st.columns(2)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    X_columns = left_column.multiselect(
        'Select Feature Columns for Logistic Regression:',
        numeric_columns,
        default=[numeric_columns[0]])
    # X_columns = st.multiselect('Select Feature Columns for Logistic Regression:', df.select_dtypes(include=['number']).columns.tolist())
    y_column = right_column.selectbox(
        'Select Target Column for Logistic Regression:',
        df.select_dtypes(include=['object']).columns.tolist())
    test_size = left_column.slider(
        'Select Test Size for Train-Test Split for Logistic Regression:', 0.1,
        0.5, 0.2)

    penalty_option = right_column.selectbox('Select Penalty Type:',
                                            ['l2', 'l1'])
    solver_option = 'saga' if penalty_option == 'l1' else 'newton-cg'

    X = df[X_columns]
    y = LabelEncoder().fit_transform(df[y_column])

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=42)
    model = LogisticRegression(penalty=penalty_option, solver=solver_option)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    st.write("Classification Report:", classification_report(y_test, y_pred))

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    st.pyplot()

    # Confusion Matrix Heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot()


# Function to perform K-Means Clustering
def perform_k_means_clustering(df):
    st.subheader("K-Means Clustering")
    # Pilih fitur numerik
    features = st.multiselect(
        'Select features for K-Means clustering:',
        df.select_dtypes(include=['number']).columns.tolist())
    if not features or len(features) < 2:
        st.warning('Please select at least two numerical features.')
        return

    X = df[features]

    # Pra-pemrosesan: Skalakan fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Metode Elbow untuk menentukan jumlah klaster optimal
    distortions = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k).fit(X_scaled)
        distortions.append(
            sum(
                np.min(cdist(X_scaled, kmeans.cluster_centers_, 'euclidean'),
                       axis=1)) / X_scaled.shape[0])

    plt.plot(range(1, 11), distortions, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    st.pyplot(plt)

    optimal_clusters = np.argmin(np.diff(np.diff(distortions))) + 2
    st.write(
        f"The optimal number of clusters based on the Elbow Method is: {optimal_clusters}"
    )

    num_clusters = st.slider(
        'Select Number of Clusters for K-Means (recommended from Elbow Method):',
        2, 10, optimal_clusters)

    # Lakukan klastering K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(X_scaled)
    df['Cluster'] = kmeans.labels_

    # Tampilkan pusat klaster dan label
    st.write('Cluster Centers (in scaled space):', kmeans.cluster_centers_)
    st.write(df)

    # Ambil statistik dari semua klaster
    cluster_stats = []
    for i in range(num_clusters):
        cluster_stat = df[df['Cluster'] == i].describe()
        cluster_stats.append(cluster_stat)

    # Analisis Klaster yang Mendetail dan Kesimpulan
    for i in range(num_clusters):
        st.write(f"Cluster {i} Statistics:")
        st.write(cluster_stats[i])

        conclusions = []
        for j in range(num_clusters):
            if i != j:
                conclusion = f"Compared to Cluster {j}, Cluster {i} has "

                # Contoh perbandingan berdasarkan rata-rata fitur pertama (gantikan dengan analisis yang relevan)
                if cluster_stats[i][features[0]]['mean'] > cluster_stats[j][
                        features[0]]['mean']:
                    conclusion += f"a higher average of {features[0]}."
                else:
                    conclusion += f"a lower average of {features[0]}."

                conclusions.append(conclusion)

        st.write(conclusions)

    # Visualisasi 2D (gunakan dua fitur pertama)
    for i in range(num_clusters):
        subset = df[df['Cluster'] == i]
        plt.scatter(subset[features[0]],
                    subset[features[1]],
                    label=f"Cluster {i}",
                    alpha=0.6)
        plt.scatter(kmeans.cluster_centers_[i][0],
                    kmeans.cluster_centers_[i][1],
                    marker='x',
                    color='red')

    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.title(f'K-Means Clustering with {num_clusters} clusters')
    st.pyplot(plt)

    # Penilaian Kualitas Klaster
    silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
    st.write('Silhouette Score:', silhouette_avg)
    for i in range(num_clusters):
        st.write(f"Cluster {i} Statistics:")
        st.write(df[df['Cluster'] == i].describe())

        # Opsional: Pairplot untuk fitur yang dipilih dalam masing-masing klaster
        sns.pairplot(df[df['Cluster'] == i][features + ['Cluster']],
                     hue='Cluster')
        st.pyplot(plt)


# Function to perform Time-Series Analysis
def perform_time_series_analysis(df):
    st.subheader("Time Series Analysis")
    time_column = st.selectbox(
        'Select Time Column:',
        df.select_dtypes(include=['datetime']).columns.tolist())
    if not time_column:  # If no features are selected
        st.warning('Please select column for time.')
        return
    target_column = st.selectbox(
        'Select Target Column for Time-Series Analysis:',
        df.select_dtypes(include=['number']).columns.tolist())
    window_size = st.slider('Select Window Size for Moving Average:', 3, 30)

    df[time_column] = pd.to_datetime(df[time_column])
    df.set_index(time_column, inplace=True)
    moving_avg = df[target_column].rolling(window=window_size).mean()

    fig, ax = plt.subplots()
    ax.plot(df[target_column], label='Original')
    ax.plot(moving_avg, label=f'Moving Average (window={window_size})')
    ax.legend()
    st.pyplot(fig)


# Function to perform Hierarchical Clustering
def perform_hierarchical_clustering(df):
    st.subheader("Hierarchical Clustering")
    # Select numerical columns or appropriate features
    X = df.select_dtypes(include=['number'])

    # Convert to float if not already
    X = X.astype(float)

    # Perform hierarchical clustering
    linkage_matrix = linkage(
        X, method='ward')  # You can choose different linkage methods

    # Check the data type
    if linkage_matrix.dtype != 'float64':
        st.error("Unexpected data type for linkage matrix")
        return

    # Plot dendrogram
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    st.pyplot(plt)


# Function to perform Text Analysis using Word Cloud
def perform_text_analysis(df):
    st.subheader("Text Analysis")
    text_column = st.selectbox(
        'Select a Text Column for Word Cloud:',
        df.select_dtypes(include=['object']).columns.tolist())
    text_data = " ".join(text for text in df[text_column])
    wordcloud = WordCloud().generate(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)


# (Jangan diubah yg ini) Ekstrak semua deskripsi statistik data
def analyze_dataframe(df):
    result = {}

    try:
        # Analisis Shape Dataframe
        shape_summary = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            # 'column_names': df.columns.tolist()
        }
        result['Shape of Data'] = shape_summary
    except Exception as e:
        pass

    try:
        # Analisis Data Numerik
        numerical_columns = df.select_dtypes(include=['number']).columns
        numerical_summary = df[numerical_columns].describe().transpose(
        ).to_dict()
        numerical_summary['skewness'] = df[numerical_columns].skew().to_dict()
        numerical_summary['kurtosis'] = df[numerical_columns].kurt().to_dict()
        result['Numerical Summary'] = numerical_summary
    except Exception as e:
        pass

    # try:
    #     # Analisis Data Kategorikal
    #     categorical_columns = df.select_dtypes(include=['object']).columns
    #     categorical_summary = {col: {
    #             'unique_categories': df[col].nunique(),
    #             'mode': df[col].mode().iloc[0],
    #             'frequency': df[col].value_counts().iloc[0]
    #         } for col in categorical_columns}
    #     result['Categorical Summary'] = categorical_summary
    # except Exception as e:
    #     pass
    try:
        # Analisis Data Kategorikal
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_summary = {}
        for col in categorical_columns:
            unique_values = df[col].nunique()
            value_counts = df[col].value_counts()
            if value_counts.nunique() == 1:  # If all frequencies are the same
                mode_value = 'No mode'
                frequency_value = 'No distinct frequency'
                top_5_values = 'No top 5 values'
            else:
                mode_value = df[col].mode().iloc[0]
                frequency_value = value_counts.iloc[0]
                top_5_values = value_counts.nlargest(
                    5).to_dict()  # Top 5 most frequent categories
            summary = {
                'unique_categories': unique_values,
                'mode': mode_value,
                'frequency': frequency_value,
                'top_5_values': top_5_values
            }
            categorical_summary[col] = summary
        result['Categorical Summary'] = categorical_summary
    except Exception as e:
        pass

    try:
        # Analisis Missing Values
        missing_values = df.isnull().sum().to_dict()
        missing_percentage = {
            col: (missing_values[col] / len(df) * 100)
            for col in df.columns
        }
        missing_summary = {
            "Missing Values": missing_values,
            "Percentage": missing_percentage
        }
        # result['Missing Values'] = missing_summary
    except Exception as e:
        pass

    try:
        # Analisis Korelasi
        correlation_matrix = df.corr().to_dict()
        result['Correlation Matrix'] = correlation_matrix
    except Exception as e:
        pass

    try:
        # Analisis Outliers
        z_scores = df[numerical_columns].apply(zscore)
        outliers = (z_scores.abs() >
                    2).sum().to_dict()  # Agregat jumlah outliers
        # result['Outliers'] = outliers
    except Exception as e:
        pass

    # try:
    #     # Agregasi Lengkap untuk Semua Kolom yang Mungkin
    #     all_aggregations = df.agg(['mean', 'median', 'sum', 'min', 'max', 'std', 'var', 'skew', 'kurt']).transpose().to_dict()
    #     result['All Possible Aggregations'] = all_aggregations
    # except Exception as e:
    #     pass
    try:
        # Agregasi Lengkap untuk Semua Kolom yang Mungkin
        all_aggregations = df.agg([
            'mean', 'median', 'sum', 'min', 'max', 'std', 'var', 'skew', 'kurt'
        ]).transpose()
        top_5_values = {}
        for metric in all_aggregations.columns:
            top_5_values[metric] = all_aggregations.nlargest(
                5, metric)[metric].to_dict()
        result['All Possible Aggregations'] = top_5_values
    except Exception as e:
        pass

    # try:
    #     # Agregasi Group By untuk Semua Kombinasi Kolom Kategorikal
    #     groupby_aggregations = {}
    #     for r in range(1, len(categorical_columns) + 1):
    #         for subset in combinations(categorical_columns, r):
    #             group_key = ', '.join(subset)
    #             group_data = df.groupby(list(subset)).agg(['mean', 'count', 'sum', 'min', 'max'])
    #             groupby_aggregations[group_key] = group_data.to_dict()
    #     result['Group By Aggregations'] = groupby_aggregations
    # except Exception as e:
    #     pass

    try:
        # Agregasi Group By untuk Semua Kombinasi Kolom Kategorikal
        groupby_aggregations = {}
        metrics = ['mean', 'count', 'sum', 'min', 'max']
        for r in range(1, len(categorical_columns) + 1):
            for subset in combinations(categorical_columns, r):
                group_key = ', '.join(subset)
                group_data = df.groupby(list(subset)).agg(metrics)
                top_5_group_values = {}
                for metric in metrics:
                    for col in group_data.columns.levels[1]:
                        col_metric = group_data[metric][col]
                        top_5_group_values[
                            f'{metric}_{col}'] = col_metric.nlargest(
                                5).to_dict()
                groupby_aggregations[group_key] = top_5_group_values
        result['Group By Aggregations'] = groupby_aggregations
    except Exception as e:
        pass

    return result


def visualize_analysis(result):
    # Visualisasi Shape
    shape_data = result['Shape of Data']
    st.write(
        f"Data memiliki {shape_data['rows']} baris dan {shape_data['columns']} kolom"
    )

    # Visualisasi Summary Numerik
    if 'Numerical Summary' in result:
        numerical_summary = result['Numerical Summary']
        for col, stats in numerical_summary.items():
            if col not in ['skewness', 'kurtosis']:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(stats.keys(), stats.values())
                st.pyplot(fig)

    # Visualisasi Missing Values
    if 'Missing Values' in result:
        missing_data = result['Missing Values']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(missing_data['Missing Values'].keys(),
               missing_data['Missing Values'].values())
        st.pyplot(fig)

    # Visualisasi Correlation Matrix
    if 'Correlation Matrix' in result:
        correlation_data = pd.DataFrame(result['Correlation Matrix'])
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(correlation_data, ax=ax)
        st.pyplot(fig)

    # Visualisasi Outliers
    if 'Outliers' in result:
        outliers_data = result['Outliers']
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(outliers_data.keys(), outliers_data.values())
        st.pyplot(fig)

    # Visualisasi All Possible Aggregations
    if 'All Possible Aggregations' in result:
        aggregations_data = result['All Possible Aggregations']
        for col, stats in aggregations_data.items():
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(stats.keys(), stats.values())
            st.pyplot(fig)

    # Visualisasi Quantiles
    if 'Quantiles' in result:
        quantiles_data = result['Quantiles']
        for col, stats in quantiles_data.items():
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(stats.keys(), stats.values())
            st.pyplot(fig)


def dtale_func(df):
    # dtale_app.JINJA2_ENV = dtale_app.JINJA2_ENV.overlay(autoescape=False)
    # dtale_app.app.jinja_env = dtale_app.JINJA2_ENV
    st.title('D-Tale Reporting')
    # Menjalankan Dtale
    d = dtale.show(df)

    # Mendapatkan URL Dtale
    dtale_url = d.main_url()

    # Menanamkan Dtale ke dalam Streamlit menggunakan iframe
    components.iframe(dtale_url, height=800)


def convert_streamlit_to_plotly(streamlit_code: str) -> str:
    # Menghapus baris yang berisi 'import streamlit'
    streamlit_code = streamlit_code.replace("import streamlit as st", "")

    # Menggantikan 'st.plotly_chart(<figure_name>)' dengan '<figure_name>.show()'
    # dan menghapus semua baris yang mengandung 'st.' kecuali 'st.plotly_chart'
    lines = streamlit_code.split("\n")
    converted_lines = []

    for line in lines:
        if "st." in line:
            if "st.plotly_chart" in line:
                fig_name = line[line.find("(") + 1:line.find(")")]
                converted_line = f"    {fig_name}.show()"
                converted_lines.append(converted_line)
        else:
            converted_lines.append(line)

    return "\n".join(converted_lines)


def convert_streamlit_to_python_seaborn(streamlit_code: str) -> str:
    # Menghapus baris yang berisi 'import streamlit'
    streamlit_code = streamlit_code.replace("import streamlit as st", "")

    # Menggantikan 'st.pyplot()' dengan 'plt.show()'
    streamlit_code = streamlit_code.replace("st.pyplot()", "plt.show()")

    # Menggantikan 'st.title("...")' dengan komentar '# ...'
    lines = streamlit_code.split("\n")
    converted_lines = []

    for line in lines:
        if "st.title(" in line:
            title_content = line[line.find("(") + 1:line.find(")")]
            converted_lines.append(f"# {title_content}")
        elif "st.subheader(" in line:
            title_content = line[line.find("(") + 1:line.find(")")]
            converted_lines.append(f"# {title_content}")
        elif "st." not in line or "st.pyplot()" in line:
            converted_lines.append(line)

    return "\n".join(converted_lines)


def display_html_files_from_dir(directory):
    """Recursively fetch and display HTML files from a directory and its subdirectories."""
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)

        if os.path.isfile(full_path) and item.endswith('.html'):
            try:
                with open(full_path, "r") as f:
                    bokeh_html = f.read()
                    st.components.v1.html(bokeh_html, width=1000, height=600)
            except Exception as e:
                st.write(f"Error reading '{full_path}': {e}")
        elif os.path.isdir(full_path):
            display_html_files_from_dir(
                full_path)  # Recursively dive into subdirectories


def autoviz_app(df):
    # Buat instance dari AutoViz
    AV = AutoViz_Class()

    # Directory to save HTML plots
    save_dir = "saved_plots"
    os.makedirs(save_dir, exist_ok=True)

    # Assuming you have df (your DataFrame) here
    AV.AutoViz("",
               sep=",",
               depVar="",
               dfte=df,
               header=0,
               verbose=1,
               lowess=False,
               chart_format="html",
               max_rows_analyzed=150000,
               max_cols_analyzed=30,
               save_plot_dir=save_dir)

    # Display all saved plots in Streamlit
    display_html_files_from_dir(save_dir)


def run_lazy_predict(X_train, X_test, y_train, y_test):
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    return models


def get_sample_data(dataset_name):
    """Mendapatkan sample dataset dari seaborn."""
    if dataset_name == 'Iris (Dummy Data)':
        return sns.load_dataset('iris')
    elif dataset_name == 'Tips (Dummy Data)':
        return sns.load_dataset('tips')
    elif dataset_name == 'Titanic (Dummy Data)':
        return sns.load_dataset('titanic')
    elif dataset_name == 'Gap Minder (Dummy Data)':
        return px.data.gapminder()
    # Tambahkan dataset lainnya sesuai kebutuhan
    else:
        return None


def handle_file_upload():
    st.session_state.uploaded = False
    # Inisialisasi API Kaggle
    # api = KaggleApi()
    # api.authenticate()

    # kaggle_username = os.environ.get("KAGGLE_USERNAME")
    # kaggle_key = os.environ.get("KAGGLE_KEY")

    # def get_kaggle_datasets():
    #     datasets = api.dataset_list(search='', file_type='csv', sort_by='hottest')
    #     dataset_names = [ds.ref for ds in datasets]
    #     return dataset_names

    # def load_kaggle_dataset(dataset_name):
    #     api.dataset_download_files(dataset_name, path='./', unzip=True)
    #     csv_file = [f for f in os.listdir() if f.endswith('.csv')][0]
    #     df = pd.read_csv(csv_file)
    #     os.remove(csv_file)  # Menghapus file CSV setelah digunakan
    #     return df

    df = pd.DataFrame()

    with st.columns([1, 2, 1])[1]:
        st.subheader('Upload your CSV / Excel data:')
        option = st.selectbox(
            'Pilih sumber data:',
            ('Upload Your File', 'Iris (Dummy Data)', 'Tips (Dummy Data)',
             'Titanic (Dummy Data)', 'Gap Minder (Dummy Data)'))

        if option == 'Upload Your File':
            file = st.file_uploader("Upload file", type=['csv', 'xls', 'xlsx'])
            if file:
                loader = LoadDataframe(file)
                df = loader.load_file_auto_delimiter()
                # try:
                # df = LoadDataframe.load_file_auto_delimiter(file)
                # df = pd.read_csv(
                #     file
                # )  # Gantikan ini dengan fungsi Anda sendiri untuk membaca file
                st.session_state.df = df
                st.session_state.uploaded = True
                st.experimental_rerun()
                # main()
                # except:
                #     st.error("Mohon masukkan file dengan format yang benar.")
        # Tambahkan logika lainnya untuk opsi lainnya di sini
        else:
            df = get_sample_data(option)
            st.session_state.df = df
            st.session_state.uploaded = True
            st.experimental_rerun()
            # main()


def main():
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    openai.api_key = st.secrets['user_api']

    df = st.session_state.df
    if st.button('Change the dataset.'):
        handle_file_upload()
    st.subheader("Your dataframe (sample 10 first rows).")
    st.dataframe(df.head(10))
    # try:
    if not df.empty:
        # st.session_state.df = df
        # st.session_state.uploaded = True
        df.to_csv("temp_file.csv", index=False)
        uploaded_file_path = "temp_file.csv"

        # st.write("Sample Top 5 Rows")
        # st.dataframe(df.head())
        dataviz = DataViz(df)

        # Hide dulu karna kayanya makan tempat banget.
        # st.write('---')
        # analytics_df.info()
        # st.write('---')
        # analytics_df.basic()
        # st.write('---')
        # # Extract df schema
        schema_dict = df.dtypes.apply(lambda x: x.name).to_dict()
        schema_str = json.dumps(schema_dict)
        # st.write("\nDataframe schema : ", schema_str)

        # # Extract the first two rows into a dictionary
        rows_dict = df.head(2).to_dict('records')
        rows_str = json.dumps(rows_dict, default=str)

        st.markdown("""
            <style>
                .stButton>button {
                    background-color: #E8E8E8;   /* Warna abu-abu muda */
                    border: 2px solid #C0C0C0;   /* Border abu-abu */
                    color: #333;                 /* Warna teks gelap */
                    padding: 8px 20px;           /* Padding lebih sedikit */
                    text-align: center;          /* Teks tengah */
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    width: 300px;
                    white-space: normal !important;
                    border-radius: 5px;          /* Border radius */
                    transition: background-color 0.3s, color 0.3s;  /* Transisi warna saat di-hover */
                }
        
                .stButton>button:hover {
                    background-color: #C0C0C0;   /* Warna saat tombol di-hover */
                    color: #FFF;                /* Warna teks saat tombol di-hover */
                }
            </style>
            """,
                    unsafe_allow_html=True)

        # st.sidebar.subheader('Pilih metode eksplorasi:')
        # # Tombol 1
        # if st.sidebar.button(
        #         '🕵️ Explore Visualization & Insight with UlikData',
        #         key='my-btn0'):
        #     st.session_state.manual_exploration = True
        #     st.session_state.auto_exploration = False
        #     st.session_state.show_analisis_lanjutan = False
        #     st.session_state.show_natural_language_exploration = False
        #     st.session_state.story_telling = False
        #     st.session_state.vizz_tools = False

        # # Tombol 4
        # # st.sidebar.markdown('<button class="my-btn">4. Natural Language (Best for Data Visualization)</button>', unsafe_allow_html=True)
        # if st.sidebar.button(
        #         '💬 Explore with Natural Language (Best for Data Visualization)',
        #         key='my-btn2'):
        #     st.session_state.manual_exploration = False
        #     st.session_state.auto_exploration = False
        #     st.session_state.show_analisis_lanjutan = False
        #     st.session_state.show_natural_language_exploration = True
        #     st.session_state.story_telling = False
        #     st.session_state.sentiment = False
        #     st.session_state.classification = False
        #     st.session_state.vizz_tools = False

        # # Tombol 5
        # # st.sidebar.markdown('<button class="my-btn">5. Auto Reporting (Best for Survey Data)</button>', unsafe_allow_html=True)
        # if st.sidebar.button(
        #         '🤖 Automatic Insight Generations (by UlikData x GPT) - Under Maintenance',
        #         key='my-btn3'):
        #     st.session_state.manual_exploration = False
        #     st.session_state.auto_exploration = False
        #     st.session_state.show_analisis_lanjutan = False
        #     st.session_state.show_natural_language_exploration = False
        #     st.session_state.story_telling = True
        #     st.session_state.sentiment = False
        #     st.session_state.classification = False
        #     st.session_state.vizz_tools = False

        # # Tombol 7
        # # st.sidebar.markdown('<button class="my-btn">6. Sentiment Classifications (Zero Shot)</button>', unsafe_allow_html=True)
        # if st.sidebar.button('🧠 Machine Learning (Classification Model)',
        #                      key='my-btn7'):
        #     st.session_state.manual_exploration = False
        #     st.session_state.auto_exploration = False
        #     st.session_state.show_analisis_lanjutan = False
        #     st.session_state.show_natural_language_exploration = False
        #     st.session_state.story_telling = False
        #     st.session_state.sentiment = False
        #     st.session_state.classification = True
        #     st.session_state.vizz_tools = False

        # # Tombol 8
        # if st.sidebar.button('Testing BI Tools (under development)',
        #                      key='my-btn8'):
        #     st.session_state.manual_exploration = False
        #     st.session_state.auto_exploration = False
        #     st.session_state.show_analisis_lanjutan = False
        #     st.session_state.show_natural_language_exploration = False
        #     st.session_state.story_telling = False
        #     st.session_state.sentiment = False
        #     st.session_state.classification = False
        #     st.session_state.vizz_tools = True

        tab_manual, tab_auto_insight, tab_reco_vizz, tab_nlp, tab_ml = st.tabs(
            [
                "1-Click Exploration", "Automatic Data Exploration",
                "Recomender Visualization", "Ask Your Data",
                "Machine Learning Modeling"
            ])

        # if st.session_state.get('manual_exploration', False):
        # with tab_manual:
        #     dataviz.visualization()

        # if st.session_state.get('show_natural_language_exploration', False):
        # if tabs == "Explore Visualization & Insight with UlikData":
        with tab_nlp:
            st.subheader("Natural Language Exploration")
            input_pengguna = ""
            style_choosen = 'Visualization'
            input_pengguna = st.text_area(
                """Masukkan perintah anda untuk mengolah data tersebut: (ex: 'Buatkan scatter plot antara kolom A dan B', 'Hitung korelasi antara semua kolom numerik')""",
                value=
                "Buatkan semua visualisasi yang mungkin dengan sedetail mungkin untuk semua case yang relevan."
            )
            style_choosen = st.selectbox('Choose an Objective:',
                                         ('Visualization', 'General Question'))

            if 'button_clicked' not in st.session_state:
                st.session_state.button_clicked = False

            button = st.button("Submit", key='btn_submit')
            if button:
                st.session_state.button_clicked = True

            if (input_pengguna != "") & (
                    input_pengguna != None) & st.session_state.button_clicked:
                st.session_state.button_clicked = True
                with st.spinner('Wait for it...'):
                    script = request_prompt(input_pengguna, uploaded_file_path,
                                            schema_str, rows_str,
                                            style_choosen, None, None, 0)
                    st.session_state['script'] = script
                    # Membuat 2 kolom utama
                    main_col1, main_col3 = st.columns(
                        [2, 1])  # kolom pertama memiliki lebar 2x kolom kedua

                    # Membuat 2 sub-kolom di dalam main_col1
                    sub_col1, sub_col2 = main_col1.columns(2)

                    with main_col1:  # Gunakan kolom utama pertama
                        # st.subheader("Visualizations")
                        if style_choosen == 'General Question':
                            st.write(script)
                        else:
                            exec(str(script))

                    # button = st.button("Print Code")

                    # if button:
                    #     # st.subheader("Streamlit Script")
                    #     # st.text(script)
                    #     st.subheader(f"{style_choosen} Script")
                    #     if style_choosen == 'Plotly':
                    #         st.text(
                    #             convert_streamlit_to_plotly(
                    #                 st.session_state['script']))
                    #     elif style_choosen == 'Seaborn':
                    #         st.text(
                    #             convert_streamlit_to_python_seaborn(
                    #                 st.session_state['script']))

                    input_pengguna = ""

        # if st.session_state.get('story_telling', False):
        # elif tabs == "Explore with Natural Language (Best for Data Visualization)":
        with tab_auto_insight:
            st.title("Automated Insights by Ulikdata")

            st.markdown("""
            <style>
                .reportview-container .markdown-text-container {
                    font-family: monospace;
                    background-color: #fafafa;
                    max-width: 700px;
                    padding-right:50px;
                    padding-left:50px;
                }
            </style>
            """,
                        unsafe_allow_html=True)

            #     st.markdown(request_story_prompt(dict_stats))
            format = st.selectbox('Choose the Output:',
                                  ['Reports', 'Visualizations', 'Dashboard'])
            if format == 'Visualizations':
                min_viz = st.selectbox('Expected number of insights:',
                                       [3, 4, 5, 6, 7, 8, 9, 10])
                # min_viz = st.slider('Expected number of insights:', min_value=3, max_value=50)

                style_choosen = 'Plotly'
                style_choosen = st.selectbox(
                    'Choose a Visualization Style:',
                    ('Plotly', 'Vega', 'Seaborn', 'Matplotlib'))
                api_model = st.selectbox('Choose LLM Model:',
                                         ('GPT4', 'GPT3.5'))

                if style_choosen == 'Matplotlib':
                    style_choosen = 'Matplotlib with clean and minimalist style'

                button = st.button("Submit", key='btn_submit2')
                if button:
                    # Membagi respons berdasarkan tanda awal dan akhir kode
                    with st.spinner(
                            'Generating insights...(it may takes 1-2 minutes)'
                    ):
                        response = request_story_prompt(
                            schema_str, rows_str, min_viz, api_model,
                            style_choosen)
                        # st.text(response)
                        # Extracting the introductions
                        # pattern = r'st.write\("Insight \d+: .+?"\)\nst.write\("(.+?)"\)'
                        # pattern = r'st.write\("Insight \d+: (.+?)"\)'
                        pattern = r'# Insight \d+: (.+?)\n'

                        introductions = re.findall(pattern, response)

                        # Printing the extracted introductions
                        # for intro in introductions:
                        #     print(intro)

                        # Saving the introductions to a list
                        introduction_list = list(introductions)
                        introduction_list = [
                            "Analyze the " + s for s in introduction_list
                        ]

                        # st.text(introduction_list)
                        # for query in introduction_list:
                        #     st.write(get_answer_csv(df, query))

                        def execute_streamlit_code_with_explanations(
                                response, introduction_list):
                            # Split kode berdasarkan st.plotly_chart()
                            code_segments = response.split('st.plotly_chart(')

                            modified_code = code_segments[
                                0]  # Bagian kode sebelum plot pertama

                            progress_bar = st.progress(0)
                            for index, segment in enumerate(code_segments[1:]):
                                # Dapatkan penjelasan untuk segment ini
                                if index < len(introduction_list):
                                    explanation = get_answer_csv(
                                        uploaded_file_path,
                                        introduction_list[index])
                                    modified_code += f'\nst.write("{explanation}")\n'

                                # Tambahkan st.plotly_chart kembali
                                modified_code += 'st.plotly_chart(' + segment

                                # Update progress bar
                                progress_percentage = (index + 1) / len(
                                    code_segments[1:])
                                progress_bar.progress(progress_percentage)

                            # st.code(modified_code)
                            # Eksekusi kode yang telah dimodifikasi
                            exec(modified_code)

                        # execute_streamlit_code_with_explanations(response, introduction_list)

                        segments = response.split("BEGIN_CODE")
                        segment_iterator = iter(segments)
                        for segment in segment_iterator:
                            # Jika ada kode dalam segmen ini
                            if "END_CODE" in segment:
                                code_end = segment.index("END_CODE")
                                code = segment[:code_end].strip()
                                explanation = segment[code_end +
                                                      len("END_CODE"):].strip(
                                                      )
                                explanation = explanation.replace('"', '\\"')

                                # Coba eksekusi kode
                                # try:
                                # st.code(code)  # Tampilkan kode dalam format kode
                                # execute_streamlit_code_with_explanations(code, introduction_list)
                                exec(code)
                                # st.write("Hasil eksekusi kode:")
                                # st.write(output)
                                # except Exception as e:
                                #     st.write("Maaf terjadi kesalahan saat mengeksekusi kode untuk insight ini. Error:")
                                #     st.write(str(e))
                                # next(segment_iterator, None)  # Lewati segmen penjelasan berikutnya
                                # continue  # Lanjut ke segmen berikutnya setelah segmen penjelasan

                                # Tampilkan teks penjelasan
                                if explanation:
                                    st.write(explanation)
                            else:
                                # Jika tidak ada kode dalam segmen ini, hanya tampilkan teks
                                st.write(segment)
                        # st.write("For Developer Maintenance Purposed (will remove)")
                        # st.text(response)
                        # st.text(introduction_list)
            elif format == 'Dashboard':

                def request_dashboard(schema_str,
                                      rows_str,
                                      min_viz,
                                      api_model,
                                      library='Matplotlib'):

                    # Versi penjelasan dan code
                    messages = [{
                        "role":
                        "system",
                        "content":
                        f"I will create a dashboard with many chart in a single figure with {library} and show it in Streamlit. Every script should start with 'BEGIN_CODE' and end with 'END_CODE'."
                    }, {
                        "role":
                        "user",
                        "content":
                        f"""Create a dashboard with many charts in a single figure with {library} from data with the schema: {schema_str}, and the first 2 sample rows as an illustration: {rows_str}.
                        My dataframe has been loaded previously, named 'df'. Use it directly; do not reload the dataframe, and do not redefine the dataframe.
                        The dashboard contains many charts with proper title.
                        Extract as many as possible charts with high quality insights.
                        Every script should start with 'BEGIN_CODE' and end with 'END_CODE'.
                        Use df directly; it's been loaded before, do not reload the df, and do not redefine the df.
                        The charts must be unique and interesting, {min_viz} most important insights from the data.
                        Do not give any explanation. Response only with script start with 'BEGIN_CODE' and end with 'END_CODE'.
                        Give the title of dashboard with st.title().
                        """
                    }]

                    if api_model == 'GPT3.5':
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            # model="gpt-4",
                            messages=messages,
                            max_tokens=3000,
                            temperature=0)
                        script = response.choices[0].message['content']
                    else:
                        response = openai.ChatCompletion.create(
                            # model="gpt-3.5-turbo",
                            model="gpt-4",
                            messages=messages,
                            max_tokens=3000,
                            temperature=0)
                        script = response.choices[0].message['content']
                    return script

                min_viz = st.selectbox('Expected number of insights:',
                                       [3, 4, 5, 6, 7, 8, 9, 10])

                api_model = st.selectbox('Choose LLM Model:',
                                         ('GPT4', 'GPT3.5'))
                library = 'Matplotlib'
                library = st.selectbox('Choose a Visualization Library:',
                                       ('Altair', 'Matplotlib'))

                if library == 'Altair':
                    library = 'Altair (hconcat max 3 charts and vconcat for the rest with the same chart size)'
                if library == 'Seaborn':
                    library = 'Seaborn (facetgrid)'
                if library == 'Matplotlib':
                    library = 'Matplotlib (subplots max 3 charts in a gird row and vertical for the rest) with minimalist style'
                button = st.button("Submit", key='btn_submit3')
                if button:
                    # Membagi respons berdasarkan tanda awal dan akhir kode
                    with st.spinner(
                            'Generating simple dashboard...(it may takes 1-2 minutes)'
                    ):
                        response = request_dashboard(schema_str, rows_str,
                                                     min_viz, api_model,
                                                     library)

                        # st.write("Cek output.")
                        # st.text(response)
                        segments = response.split("BEGIN_CODE")
                        segment_iterator = iter(segments)
                        for segment in segment_iterator:
                            # Jika ada kode dalam segmen ini
                            if "END_CODE" in segment:
                                code_end = segment.index("END_CODE")
                                code = segment[:code_end].strip()
                                explanation = segment[code_end +
                                                      len("END_CODE"):].strip(
                                                      )
                                explanation = explanation.replace('"', '\\"')
                                # st.write("Cek code.")
                                # st.text(code)
                                exec(code)
                                # Tampilkan teks penjelasan
                                if explanation:
                                    st.write(explanation)
                            else:
                                # Jika tidak ada kode dalam segmen ini, hanya tampilkan teks
                                st.write(segment)

            else:

                def request_summary_points(schema_str,
                                           rows_str,
                                           api_model,
                                           context=''):

                    # Versi penjelasan dan code
                    # messages = [{
                    #     "role":
                    #     "system",
                    #     "content":
                    #     f"I will create points for you in the form of analysis to be saved in global variable string named point_summary. Every script should start with 'BEGIN_CODE' and end with 'END_CODE'."
                    # }, {
                    #     "role":
                    #     "user",
                    #     "content":
                    #     f"""Create points in the form of insights that are insightful from data with the schema: {schema_str}, and the first 2 sample rows as an illustration: {rows_str}.
                    #     My dataframe has been loaded previously, named 'df'. Use it directly; do not reload the dataframe, and do not redefine the dataframe.
                    #     Every script should start with 'BEGIN_CODE' and end with 'END_CODE'.
                    #     Use df directly; it's been loaded before, do not reload the df, and do not redefine the df.
                    #     Create insight points whose values are extracted from the dataframe df with schema: {schema_str}, then turn them into variables, and display them as insight points in Streamlit.
                    #     Write as many insights as possible that can be extracted in the form of bullet points.
                    #     """
                    # }]

                    messages = [{
                        "role":
                        "system",
                        "content":
                        f"I will create python code to generate insight points for you in the form of analysis to be saved in string st.session_state.point_summary"
                    }, {
                        "role":
                        "user",
                        "content":
                        f"""Create python code to generate insight points in the form of insights that are insightful from dataframe df with the schema: {schema_str}, and the first 2 sample rows as an illustration: {rows_str}.
                        My dataframe has been loaded previously, named 'df'. Use it directly; do not reload the dataframe, and do not redefine the dataframe.
                        Use df directly; it's been loaded before, do not reload the df, and do not redefine the df.
                        Create insight points whose values are extracted from the dataframe df with schema: {schema_str}, then turn them into variables, and saved all insights in string st.session_state.point_summary.
                        Write as many insights as possible that can be extracted in the form of bullet points. Minimal 25 insights.
                        Only response with python code. Do not respond with anything other than Python code.
                        The value in the string st.session_state.point_summary should already be in the form of a value, not a variable.
                        Define first 'point_summary' in st.session_state as empty string.
                        {context}
                        """
                    }]

                    if api_model == 'GPT3.5':
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            # model="gpt-4",
                            messages=messages,
                            max_tokens=3000,
                            temperature=0)
                        script = response.choices[0].message['content']
                    else:
                        response = openai.ChatCompletion.create(
                            # model="gpt-3.5-turbo",
                            model="gpt-4",
                            messages=messages,
                            max_tokens=3000,
                            temperature=0)
                        script = response.choices[0].message['content']

                    return script

                def request_explain_summary(text_summary, api_model):

                    # Versi penjelasan dan code
                    messages = [{
                        "role":
                        "system",
                        "content":
                        f"I will create points for you in the form of analysis to be save as string. Every script should start with 'BEGIN_CODE' and end with 'END_CODE'."
                    }, {
                        "role":
                        "user",
                        "content":
                        f"""Create points in the form of insights that are insightful from data with the schema: {schema_str}, and the first 2 sample rows as an illustration: {rows_str}.
                        My dataframe has been loaded previously, named 'df'. Use it directly; do not reload the dataframe, and do not redefine the dataframe.
                        Every script should start with 'BEGIN_CODE' and end with 'END_CODE'.
                        Use df directly; it's been loaded before, do not reload the df, and do not redefine the df.
                        Create insight points whose values are extracted from the dataframe df with schema: {schema_str}, then turn them into variables, and save them as string.
                        Write as many insights as possible with various aggregated value that can be extracted in the form of bullet points.
                        """
                    }]

                    if api_model == 'GPT3.5':
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            # model="gpt-4",
                            messages=messages,
                            max_tokens=3000,
                            temperature=0)
                        script = response.choices[0].message['content']
                    else:
                        response = openai.ChatCompletion.create(
                            # model="gpt-3.5-turbo",
                            model="gpt-4",
                            messages=messages,
                            max_tokens=3000,
                            temperature=0)
                        script = response.choices[0].message['content']

                    return script

                def request_summary_wording(text_summary, language,
                                            style_choosen, objective, format,
                                            api_model, context):
                    messages = [{
                        "role":
                        "system",
                        "content":
                        f"Aku akan menjabarkan summary kamu dengan menggunakan bahasa {language}. Dalam format {format}."
                    }, {
                        "role":
                        "user",
                        "content":
                        f"""Buatkan laporan yang insightful dengan gaya {style_choosen} dan {objective}, menggunakan bahasa {language}, dalam format {format}, serta berikan opinimu dari informasi umum yang diketahui untuk setiap point dari informasi berikut: {text_summary}. Buang insight yang tidak penting, fokus pada insight yang insightful. Tulis dalam 3000 kata. Beri Judul (dengan #) dan Subjudul (dengan ###) sesuai insight. {context}"""
                    }]

                    if api_model == 'GPT3.5':
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo-16k",
                            # model="gpt-4",
                            messages=messages,
                            max_tokens=3000,
                            temperature=0.6)
                        script = response.choices[0].message['content']
                    else:
                        response = openai.ChatCompletion.create(
                            # model="gpt-3.5-turbo",
                            model="gpt-4",
                            messages=messages,
                            max_tokens=3000,
                            temperature=0.6)
                        script = response.choices[0].message['content']

                    return script

                def split_text_into_lines(text, words_per_line=20):
                    words = text.split()
                    lines = []
                    for i in range(0, len(words), words_per_line):
                        line = ' '.join(words[i:i + words_per_line])
                        lines.append(line)
                    return '\n'.join(lines)

                api_model = st.selectbox('Choose LLM Model:',
                                         ('GPT4', 'GPT3.5'))
                language = st.selectbox('Choose Language:',
                                        ('Indonesia', 'English', 'Sunda'),
                                        key='btn_lang')
                style_choosen = st.selectbox('Choose the Formality:',
                                             ('Formal', 'Non-Formal'),
                                             key='btn_style')
                objective = st.selectbox(
                    'Choose the Style:',
                    ('Narative', 'Persuasive', 'Descriptive', 'Argumentative',
                     'Satire'),
                    key='btn_obj')
                format = st.selectbox('Choose the Format:',
                                      ('Paragraf', 'Youtube Script', 'Thread',
                                       'Caption Instagram'),
                                      key='btn_format')

                context_user = st.text_area(
                    "Berikan context untuk fokus pada analisis tertentu jika perlu. Kosongkan untuk analisis secara umum.",
                    value='')

                button = st.button("Submit", key='btn_submit2')
                if button:
                    # Membagi respons berdasarkan tanda awal dan akhir kode
                    with st.spinner(
                            'Generating insights...(it may takes 1-2 minutes)'
                    ):
                        response = request_summary_points(
                            schema_str, rows_str, api_model, context_user)

                    # st.write('Original Response: ')
                    # st.text(response)
                    # segments = response.split("BEGIN_CODE")
                    segments = response.split("```python")

                    segment_iterator = iter(segments)

                    # st.write('Displayed Response: ')
                    for segment in segment_iterator:
                        # Jika ada kode dalam segmen ini
                        if "```" in segment:
                            # code_end = segment.index("END_CODE")
                            code_end = segment.index("```")
                            code = segment[:code_end].strip()
                            explanation = segment[code_end +
                                                  # len("END_CODE"):].strip()
                                                  len("```"):].strip()
                            explanation = explanation.replace('"', '\\"')
                            # st.write('The Code to Execute: ')
                            st.text(code)
                            exec(code)

                            # Tampilkan teks penjelasan
                            # if explanation:
                            #     st.write(explanation)
                        # else:
                        #     # Jika tidak ada kode dalam segmen ini, hanya tampilkan teks
                        #     st.write(segment)

                        # text_summary = segment

                    # exec(response)

                    # st.write("Display point_summary state:")
                    # st.text(st.session_state.point_summary)

                    with st.spinner(
                            'Creating the paragraph...(it may takes 1-2 minutes)'
                    ):
                        paragraph = request_summary_wording(
                            st.session_state.point_summary, language,
                            style_choosen, objective, format, api_model,
                            context_user)
                    # st.text(split_text_into_lines(response))
                    st.write(paragraph)

                    st.write("Details:")
                    st.text(st.session_state.point_summary)

        # if st.session_state.get('classification', False):
        # elif tabs == "Machine Learning (Classification Model)":
        with tab_ml:
            st.title("Machine Learning Modeling")
            # Data cleansing options
            # Check for missing values
            if df.isnull().sum().sum() > 0:
                st.warning("Warning: Dataset contains missing values!")
                missing_stats = pd.DataFrame(df.isnull().sum(),
                                             columns=["Missing Values"])
                missing_stats["Percentage"] = (
                    missing_stats["Missing Values"] / len(df)) * 100
                st.write(missing_stats[missing_stats["Missing Values"] > 0])

                # Data cleansing options for each column
                st.subheader("Data Cleansing Options for Each Column")

                for col in df.columns:
                    if df[col].isnull().sum() > 0:
                        st.markdown(f"### {col}")

                        if df[col].dtype in ['int64', 'float64']:
                            # Numerical column
                            missing_option = st.selectbox(
                                f"How to handle missing values for {col}?", [
                                    "Do Nothing", "Drop Rows",
                                    "Fill with Mean", "Fill with Median",
                                    "Fill with Mode"
                                ])
                            if missing_option == "Drop Rows":
                                df.dropna(subset=[col], inplace=True)
                            elif missing_option == "Fill with Mean":
                                df[col].fillna(df[col].mean(), inplace=True)
                            elif missing_option == "Fill with Median":
                                df[col].fillna(df[col].median(), inplace=True)
                            elif missing_option == "Fill with Mode":
                                df[col].fillna(df[col].mode().iloc[0],
                                               inplace=True)
                        else:
                            # Categorical column
                            missing_option = st.selectbox(
                                f"How to handle missing values for {col}?",
                                ["Do Nothing", "Drop Rows", "Fill with Mode"])
                            if missing_option == "Drop Rows":
                                df.dropna(subset=[col], inplace=True)
                            elif missing_option == "Fill with Mode":
                                df[col].fillna(df[col].mode().iloc[0],
                                               inplace=True)

            else:
                st.success("Data tidak mengandung missing values.")

            # One-hot encoding for categorical columns
            one_hot_encode = st.checkbox("One-hot encode categorical columns?")
            if one_hot_encode:
                df = pd.get_dummies(df, drop_first=True)

            # Normalize data
            normalize = st.checkbox("Normalize data?")
            if normalize:
                normalization_method = st.selectbox(
                    "Choose normalization method:",
                    ["Min-Max Scaling", "Z-score Normalization"])

                numerical_cols = df.select_dtypes(
                    include=['int64', 'float64']).columns
                cols_to_normalize = st.multiselect(
                    "Select numerical columns to normalize:",
                    options=numerical_cols,
                    default=list(numerical_cols))

                if normalization_method == "Min-Max Scaling":
                    scaler = MinMaxScaler()
                    df[cols_to_normalize] = scaler.fit_transform(
                        df[cols_to_normalize])
                elif normalization_method == "Z-score Normalization":
                    mean = df[cols_to_normalize].mean()
                    std = df[cols_to_normalize].std()
                    df[cols_to_normalize] = (df[cols_to_normalize] -
                                             mean) / std

            st.write(df.head())

            # Select target column
            target_column = st.selectbox("Select the target column",
                                         df.columns)

            # Select feature columns
            # numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            feature_columns = st.multiselect(
                "Select the feature columns",
                df.select_dtypes(include=['int64', 'float64']).columns,
                default=df.select_dtypes(
                    include=['int64', 'float64']).columns[df.select_dtypes(
                        include=['int64', 'float64']).columns != target_column]
                .tolist())
            features = df[feature_columns]
            target = df[target_column]

            # Split dataset
            test_size = st.slider("Select test size (fraction)", 0.1, 0.9,
                                  0.25, 0.05)
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_size, random_state=42)

            # Select models for LazyPredict
            # available_classifiers = [
            #     "LinearSVC", "SGDClassifier", "MLPClassifier", "Perceptron", "LogisticRegression",
            #     "LogisticRegressionCV", "SVC", "CalibratedClassifierCV", "PassiveAggressiveClassifier",
            #     "LabelPropagation", "LabelSpreading", "RandomForestClassifier", "GradientBoostingClassifier",
            #     "QuadraticDiscriminantAnalysis", "HistGradientBoostingClassifier", "RidgeClassifierCV",
            #     "RidgeClassifier", "AdaBoostClassifier", "ExtraTreesClassifier", "KNeighborsClassifier",
            #     "BaggingClassifier", "BernoulliNB", "LinearDiscriminantAnalysis", "GaussianNB", "NuSVC",
            #     "DecisionTreeClassifier", "NearestCentroid", "ExtraTreeClassifier", "CheckingClassifier", "DummyClassifier"
            # ]

            # selected_classifiers = st.multiselect("Select classification models to compare:", options=available_classifiers, default=['GaussianNB','LogisticRegression','DecisionTreeClassifier','RandomForestClassifier','GradientBoostingClassifier','AdaBoostClassifier'])

            # Run LazyPredict
            if st.button("Run LazyPredict", key='btn_run_lazy_predict'):
                with st.spinner('Running LazyPredict...'):
                    results = run_lazy_predict(X_train, X_test, y_train,
                                               y_test)
                    st.write(results)

        # if st.session_state.get('vizz_tools', False):
        # elif tabs == "Testing BI Tools (under development)":
        # with tab_reco_vizz:
        #     st.subheader("Recomender Visualization")
        #     # Tableau 10 color palette
        #     tableau_10 = [
        #         '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
        #         '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF'
        #     ]

        #     # Load your DataFrame here
        #     data_used = df.copy()
        #     numeric_cols = data_used.select_dtypes(include=['number']).columns
        #     data_used[numeric_cols] = data_used[numeric_cols].fillna(0)

        #     object_cols = data_used.select_dtypes(include=['object']).columns
        #     data_used[object_cols] = data_used[object_cols].fillna(
        #         'Missing Data')

        #     # Layout
        #     column1, column2 = st.columns([1, 3])
        #     # Sidebar for DataFrame column selection
        #     with column1:
        #         st.subheader("Toolkit!")
        #         with st.expander("Select Columns"):
        #             selected_columns = []
        #             for col in data_used.columns:
        #                 prefix = '09' if data_used[
        #                     col].dtype != 'object' else 'AZ'
        #                 display_name = f"{prefix} - {col}"
        #                 if st.checkbox(display_name, False):
        #                     selected_columns.append(col)

        #         # Sidebar for DataFrame column filtering and aggregation
        #         with st.expander("Filter and Aggregate"):
        #             if selected_columns:
        #                 filter_values = {}
        #                 for col in selected_columns:
        #                     if data_used[col].dtype == 'object':
        #                         filter_values[col] = st.text_input(
        #                             f"Filter {col} (text)")
        #                     else:
        #                         data_used[col] = data_used[col].cat.as_ordered()
        #                         min_val, max_val = st.slider(
        #                             f"Filter {col} (range)",
        #                             float(data_used[col].min()),
        #                             float(data_used[col].max()),
        #                             (float(data_used[col].min()),
        #                              float(data_used[col].max())))
        #                         filter_values[col] = (min_val, max_val)
        #                 if len(selected_columns) > 1:
        #                     aggregation = st.selectbox(
        #                         "Aggregation",
        #                         ["sum", "min", "max", "mean", "median"])
        #                 else:
        #                     aggregation = "count"

        #         with st.expander("Data Selection Customizer"):
        #             if len(selected_columns) > 1 and len(selected_columns) < 4:
        #                 customizer = {}
        #                 customizer['X'] = st.selectbox(
        #                     "Select X Variable:",
        #                     [x for x in selected_columns])
        #                 customizer['Y'] = st.selectbox(
        #                     "Select Y Variable:",
        #                     [x for x in selected_columns])
        #                 if len(selected_columns) == 3:
        #                     customizer['Z'] = st.selectbox(
        #                         "Select Z Variable:",
        #                         [x for x in selected_columns])

        #     with column2:
        #         insight = None
        #         column2.subheader("Visualization")
        #         if len(selected_columns) == 1:
        #             col = selected_columns[0]
        #             fig = None
        #             if data_used[col].dtype == 'object':
        #                 chart_type = column2.selectbox(
        #                     "Select Type of Chart", [
        #                         "Bar Chart", "Columns Chart", "Pie Chart",
        #                         "Doughnut Chart"
        #                     ])
        #                 if chart_type == "Bar Chart":
        #                     fig, insight = BarChart(data_used, col, tableau_10,
        #                                             aggregation)

        #                 elif chart_type == "Columns Chart":
        #                     fig, insight = BarChart(data_used,
        #                                             col,
        #                                             tableau_10,
        #                                             aggregation,
        #                                             columns_chart=True)
        #                 elif chart_type == "Pie Chart":
        #                     fig, insight = PieChart(data_used, col, tableau_10,
        #                                             aggregation)
        #                 elif chart_type == "Doughnut Chart":
        #                     fig, insight = PieChart(data_used,
        #                                             col,
        #                                             tableau_10,
        #                                             aggregation,
        #                                             donut=True)
        #             else:
        #                 chart_type = column2.selectbox(
        #                     "Select Type of Chart", ["Histogram", "Boxplot"])
        #                 if chart_type == "Histogram":
        #                     fig = px.histogram(
        #                         data_used,
        #                         x=col,
        #                         color_discrete_sequence=tableau_10)
        #                     insight = "No"
        #                 elif chart_type == "Boxplot":
        #                     fig = px.box(data_used,
        #                                  x=col,
        #                                  color_discrete_sequence=tableau_10)
        #                     insight = "No"

        #             column2.plotly_chart(fig, use_container_width=True)
        #             column2.write('---')
        #             column2.markdown("#### :blue[Insight!]")
        #             column2.write(insight)

        #         elif len(selected_columns) == 2:
        #             col1 = selected_columns[0]
        #             col2 = selected_columns[1]
        #             if data_used[col1].dtype == 'object' and data_used[
        #                     col2].dtype == 'object':
        #                 fig = px.bar(data_used,
        #                              x=col1,
        #                              color=col2,
        #                              color_discrete_sequence=tableau_10)
        #                 insight = "No"

        #             elif data_used[col1].dtype != 'object' and data_used[
        #                     col2].dtype != 'object':
        #                 fig = px.scatter(data_used,
        #                                  x=col1,
        #                                  y=col2,
        #                                  color_discrete_sequence=tableau_10)
        #                 insight = "No"
        #             elif data_used[col1].dtype == 'object' and data_used[
        #                     col2].dtype != 'object':
        #                 fig, insight = BarChart(data_used,
        #                                         col1,
        #                                         tableau_10,
        #                                         aggregation,
        #                                         col2=col2)
        #             else:
        #                 fig, insight = BarChart(data_used,
        #                                         col2,
        #                                         tableau_10,
        #                                         aggregation,
        #                                         col2=col1)
        #             column2.plotly_chart(fig, use_container_width=True)
        #             column2.write('---')
        #             column2.markdown("#### :blue[Insight!]")
        #             column2.write(insight)

        #         elif len(selected_columns) == 3:
        #             # Membuat scatter plot 3D dengan ukuran dan garis pinggir
        #             fig = go.Figure(data=[
        #                 go.Scatter3d(
        #                     x=data_used[selected_columns[0]],
        #                     y=data_used[selected_columns[1]],
        #                     z=data_used[selected_columns[2]],
        #                     mode='markers',
        #                     marker=dict(
        #                         size=6,
        #                         color=
        #                         'blue',  # set color to an array/list of desired values
        #                         opacity=0.8,
        #                         line=dict(color='black', width=2)))
        #             ])

        #             fig.update_layout(width=800, height=800)

        #             # Menampilkan plot di Streamlit
        #             st.plotly_chart(fig)

        #         elif len(selected_columns) >= 4:
        #             cat_cols = data_used[selected_columns].select_dtypes(
        #                 include='object').columns.tolist()
        #             if len(cat_cols) > 0:
        #                 data = data_used[selected_columns].groupby(
        #                     by=cat_cols).agg(aggregation)
        #             else:
        #                 data = data_used[selected_columns].copy()
        #             column2.dataframe(data, use_container_width=True)
            # Sidebar kiri atas: Radio button untuk memilih kolom
            # selected_columns = st.sidebar.multiselect("Select columns to visualize", df.columns)
            # Inisialisasi state jika belum ada
            # if 'selected_columns' not in st.session_state:
            #     st.session_state.selected_columns = []

            # # Multiselect di tengah untuk memilih kolom
            # st.session_state.selected_columns = st.multiselect(
            #     "Select Columns",
            #     df.columns.tolist(),
            #     default=st.session_state.selected_columns
            # )

            # # Pilihan agregasi
            # if len(st.session_state.selected_columns) > 0:
            #     numeric_cols = [col for col in st.session_state.selected_columns if pd.api.types.is_numeric_dtype(df[col])]
            #     if len(numeric_cols) > 0:
            #         aggregation = st.selectbox("Select aggregation for numeric columns", ["sum", "count", "mean", "median"])

            # # Main content
            # st.title('Data Visualization')

            # # Fungsi untuk mendeteksi tipe data
            # def detect_dtype(column):
            #     return 'numeric' if pd.api.types.is_numeric_dtype(df[column]) else 'categorical'

            # # Visualisasi sesuai dengan jumlah dan tipe kolom yang dipilih
            # if len(st.session_state.selected_columns) == 1:
            #     col = st.session_state.selected_columns[0]
            #     if detect_dtype(col) == 'numeric':
            #         st.write(px.histogram(df, x=col))
            #     else:
            #         st.write(px.bar(df, x=col))

            # elif len(st.session_state.selected_columns) == 2:
            #     col1, col2 = st.session_state.selected_columns
            #     if detect_dtype(col1) == 'numeric' and detect_dtype(col2) == 'numeric':
            #         st.write(px.scatter(df, x=col1, y=col2))
            #     else:
            #         st.write(px.bar(df, x=col1, y=col2))

            # elif len(st.session_state.selected_columns) == 3:
            #     col1, col2, col3 = st.session_state.selected_columns
            #     # Anda bisa menambahkan logika lain di sini
            #     st.write(px.scatter_3d(df, x=col1, y=col2, z=col3))

            # elif len(st.session_state.selected_columns) == 4:
            #     col1, col2, col3, col4 = st.session_state.selected_columns

            #     # Mendeteksi tipe data untuk setiap kolom
            #     types = [detect_dtype(col) for col in st.session_state.selected_columns]

            #     if types.count('numeric') == 4:
            #         # Semua kolom numerik: gunakan kolom ke-4 sebagai ukuran poin ('size')
            #         st.write(px.scatter(df, x=col1, y=col2, color=col3, size=col4))

            #     elif types.count('categorical') >= 1:
            #         # Ada setidaknya satu kolom kategorikal: gunakan sebagai 'hue'
            #         categorical_col = [col for col, dtype in zip(st.session_state.selected_columns, types) if dtype == 'categorical'][0]
            #         numeric_cols = [col for col in st.session_state.selected_columns if col != categorical_col]
            #         st.write(px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=categorical_col, size=numeric_cols[2]))

            #     else:
            #         st.write("Visualisasi untuk kombinasi ini belum diimplementasikan.")

            # else:
            #     st.write("Displaying raw data:")
            #     st.write(df[st.session_state.selected_columns])

        # if st.session_state.get('sentiment', False):
        #     with st.spinner('Downloading the pretrained model...'):
        #         # Load BART model (Pastikan Anda memiliki model yang sesuai untuk sentiment analysis)
        #         tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        #         model = BartForSequenceClassification.from_pretrained('facebook/bart-large')

        #     def classify_sentiment(text):
        #         inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        #         outputs = model(**inputs)
        #         sentiment_id = outputs.logits.argmax(-1).item()

        #         if sentiment_id == 0:
        #             return "Negative"
        #         elif sentiment_id == 1:
        #             return "Neutral"
        #         else:
        #             return "Positive"

        #     def classify_with_progress(df, column):
        #         # Inisialisasi progress bar
        #         progress_bar = st.progress(0)
        #         total = len(df)
        #         sentiments = []

        #         for index, row in df.iterrows():
        #             sentiment = classify_sentiment(row[column])
        #             sentiments.append(sentiment)

        #             # Perbarui progress bar
        #             progress = (index + 1) / total
        #             progress_bar.progress(progress)

        #         # Selesai
        #         st.write("Classification complete!")
        #         return sentiments

        #     st.title('Sentiment Analysis with BART')
        #     # Choose a column
        #     column = st.selectbox('Select a column for sentiment analysis', df.columns)
        #     button = st.button("Submit")
        #     if button:
        #         # Classify sentiment
        #         with st.spinner('Classifying sentiments...(takes 5-10 minutes, please wait).'):
        #             df['Sentiment_by_BART'] = classify_with_progress(df, column)

        #         # Display output
        #         st.write(df[[column, 'Sentiment_by_BART']])

        #     # Download the output as CSV
        #     st.download_button("Download CSV with sentiments", df.to_csv(index=False), "sentiments.csv", "text/csv")


if __name__ == '__main__':
    # # Inisialisasi session state untuk df dan status upload jika belum ada
    # if 'df' not in st.session_state:
    #     st.session_state.df = pd.DataFrame()
    # if 'uploaded' not in st.session_state:
    #     st.session_state.uploaded = False

    # # Jika DataFrame belum di-upload, tampilkan menu upload
    # if st.session_state.uploaded:
    #     main()
    # else:
    #     handle_file_upload()

    # Cek apakah user sudah login atau belum
    if 'logged' not in st.session_state:
        st.session_state.logged = False

    if st.session_state.logged:
        # Inisialisasi session state untuk df dan status upload jika belum ada
        if 'df' not in st.session_state:
            st.session_state.df = pd.DataFrame()
        if 'uploaded' not in st.session_state:
            st.session_state.uploaded = False

        # Jika DataFrame belum di-upload, tampilkan menu upload
        if st.session_state.uploaded:
            main()
        else:
            handle_file_upload()
    else:
        login()
