import numpy as np
# import langchain
import pandas as pd
import missingno as msno
import io
import sys
from ydata_profiling import ProfileReport
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.imputation import mice
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn import svm
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import json
import base64
import plotly.io as pio
from bs4 import BeautifulSoup
from PIL import Image
from scipy import stats
import lifelines
from lifelines import KaplanMeierFitter, CoxPHFitter
from prompts  import *
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import openai
from openai import OpenAI
from tableone import TableOne
from scipy import stats
from streamlit_chat import message
import random
from random import randint
import os
from sklearn import linear_model
import statsmodels.api as sm
import category_encoders as ce
from mpl_toolkits.mplot3d import Axes3D
import shap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



st.set_page_config(page_title='AutoAnalyzer', layout = 'centered', page_icon = ':chart_with_upwards_trend:', initial_sidebar_state = 'auto')
# if st.button('Click to toggle sidebar state'):
#     st.session_state.sidebar_state = 'collapsed' if st.session_state.sidebar_state == 'expanded' else 'expanded'
#     # Force an app rerun after switching the sidebar state.
#     st.experimental_rerun()
    
#     # Initialize a session state variable that tracks the sidebar state (either 'expanded' or 'collapsed').
if 'last_response' not in st.session_state:
     st.session_state.last_response = ''
     
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

if 'modified_df' not in st.session_state:
    st.session_state.modified_df = pd.DataFrame()
    
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ''
    
if "gen_csv" not in st.session_state:
    st.session_state.gen_csv = None
    
if "df_to_download" not in st.session_state:
    st.session_state.df_to_download = None
    

@st.cache_data
def is_valid_api_key(api_key):
    openai.api_key = api_key

    try:
        # Send a test request to the OpenAI API
        response = openai.Completion.create(model="text-davinci-003",                     
                    prompt="Hello world")['choices'][0]['text']
        return True
    except Exception:
        pass

    return False


def is_bytes_like(obj):
    return isinstance(obj, (bytes, bytearray, memoryview))


def save_image(plot, filename):
    if is_bytes_like(plot):        
        img = io.BytesIO(plot)
    else:
        img = io.BytesIO()
        plot.savefig(img, format='png')
    btn = st.download_button(
        label="Download your plot.",
        data = img,
        file_name=filename,
        mime='image/png',
    )
    
@st.cache_data    
def generate_regression_equation(intercept, coef, x_col):
    equation = f"y = {round(intercept,4)}"

    for c, feature in zip(coef, x_col):
        equation += f" + {round(c,4)} * {feature}"

    return equation


def df_download_options(df, report_type, format):

    file_name = f'{report_type}.{format}'

    if format == 'csv':
        data = df.to_csv(index=True)
        mime = 'text/csv'
    if format == 'json':
        data = df.to_json(orient='records')
        mime = 'application/json'
    if format == 'html':
        data = df.to_html()
        mime = 'text/html'
    if True:
        st.download_button(
            label="Download your report.",
            data=data,
            # data=df.to_csv(index=True),
            file_name=file_name,
            mime=mime,
        )

@st.cache_data
def plot_mult_linear_reg(df, x, y):
    # with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    # st.write('Intercept: \n', regr.intercept_)
    # st.write('Coefficients: \n', regr.coef_)

    # with statsmodels
    x = sm.add_constant(x) # adding a constant
    
    model = sm.OLS(y, x).fit()
    predictions = model.predict(x) 
    
    print_model = model.summary2()
    st.write(print_model)
    try:
        df_mlr_output = print_model.tables[1]
    except:
        st.write("couldn't generate dataframe version")
    return print_model, df_mlr_output, regr.intercept_, regr.coef_
    
@st.cache_data   
def all_categorical(df):
    categ_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype != 'object']
    filtered_categorical_cols = [col for col in categ_cols if df[col].nunique() <= 15]
    all_categ = filtered_categorical_cols + numeric_cols
    return all_categ

@st.cache_data
def all_numerical(df):
    numerical_cols = df.select_dtypes(include='number').columns.tolist()

    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() == 2:
            unique_values = df[col].unique()
            if 0 in unique_values and 1 in unique_values:
                continue

            value_counts = df[col].value_counts()
            most_frequent_value = value_counts.idxmax()
            least_frequent_value = value_counts.idxmin()

            if most_frequent_value != 0 and least_frequent_value != 1:
                df[col] = np.where(df[col] == most_frequent_value, 0, 1)
                st.write(f"Replaced most frequent value '{most_frequent_value}' with 0 and least frequent value '{least_frequent_value}' with 1 in column '{col}'.")
                numerical_cols.append(col)  # Update numerical_cols

    return numerical_cols


def filter_dataframe(df):
    # Get the column names and data types of the dataframe
    columns = df.columns
    dtypes = df.dtypes

    # Create a sidebar for selecting columns to exclude
    excluded_columns = st.multiselect("Exclude Columns", columns)

    # Create a copy of the dataframe to apply the filters
    filtered_df = df.copy()

    # Exclude the selected columns from the dataframe
    filtered_df = filtered_df.drop(excluded_columns, axis=1)

    # Get the column names and data types of the filtered dataframe
    filtered_columns = filtered_df.columns
    filtered_dtypes = filtered_df.dtypes

    # Create a sidebar for selecting numerical variables and their range
    numerical_columns = [col for col, dtype in zip(filtered_columns, filtered_dtypes) if dtype in ['int64', 'float64']]
    for col in numerical_columns:
        min_val = filtered_df[col].min()
        max_val = filtered_df[col].max()
        st.write(f"**{col}**")
        min_range, max_range = st.slider("", min_val, max_val, (min_val, max_val), key=col)

        # Filter the dataframe based on the selected range
        if min_range > min_val or max_range < max_val:
            filtered_df = filtered_df[(filtered_df[col] >= min_range) & (filtered_df[col] <= max_range)]

    # Create a sidebar for selecting categorical variables and their values
    categorical_columns = [col for col, dtype in zip(filtered_columns, filtered_dtypes) if dtype == 'object']
    for col in categorical_columns:
        unique_values = filtered_df[col].unique()
        selected_values = st.multiselect(col, unique_values, unique_values)

        # Filter the dataframe based on the selected values
        if len(selected_values) < len(unique_values):
            filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

    return filtered_df


# Function to generate a download link
@st.cache_data
def get_download_link(file_path, file_type):
    with open(file_path, "rb") as file:
        contents = file.read()
    base64_data = base64.b64encode(contents).decode("utf-8")
    download_link = f'<a href="data:application/octet-stream;base64,{base64_data}" download="tableone_results.{file_type}">Click here to download the TableOne results in {file_type} format.</a>'
    return download_link

@st.cache_data
def find_binary_categorical_variables(df):
    binary_categorical_vars = []
    for col in df.columns:
        unique_values = df[col].unique()
        if len(unique_values) == 2:
            binary_categorical_vars.append(col)
    return binary_categorical_vars

@st.cache_data
def calculate_odds_older(table):
    odds_cases = table.iloc[1, 1] / table.iloc[1, 0]
    odds_controls = table.iloc[0, 1] / table.iloc[0, 0]
    odds_ratio = odds_cases / odds_controls
    return odds_cases, odds_controls, odds_ratio

@st.cache_data
def calculate_odds(table):
    odds_cases = table.iloc[1, 1] / table.iloc[1, 0]
    odds_controls = table.iloc[0, 1] / table.iloc[0, 0]
    odds_ratio = odds_cases / odds_controls
    return odds_cases, odds_controls, odds_ratio

@st.cache_data
def generate_2x2_table(df, var1, var2):
    table = pd.crosstab(df[var1], df[var2], margins=True)
    table.columns = ['No ' + var2, 'Yes ' + var2, 'Total']
    table.index = ['No ' + var1, 'Yes ' + var1, 'Total']
    return table

@st.cache_data
def plot_survival_curve(df, time_col, event_col):
    # Create a Kaplan-Meier fitter object
    try:
        kmf = KaplanMeierFitter()

        # Fit the survival curve using the dataframe
        kmf.fit(df[time_col], event_observed=df[event_col])

        # Plot the survival curve
        fig, ax = plt.subplots()
        kmf.plot(ax=ax)

        # Add labels and title to the plot
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Survival Curve')

        # Display the plot
        st.pyplot(fig)
        return fig
    except TypeError:
        st.warning("Find the right columns for time and event.")
    

        

@st.cache_data
def calculate_rr_arr_nnt(tn, fp, fn, tp):
    rr = (tp / (tp + fn)) / (fp / (fp + tn)) if fp + tn > 0 and tp + fn > 0 else np.inf
    arr = (fn / (fn + tp)) - (fp / (fp + tn)) if fn + tp > 0 and fp + tn > 0 else np.inf
    nnt = 1 / arr if arr > 0 else np.inf
    return rr, arr, nnt



def fetch_api_key():
    # Try to get the API key from an environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    # If the API key is found in the environment variables, return it
    if api_key:
        return api_key

    # If the API key is not found, check if it's already in the session state
    if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
        return st.session_state.openai_api_key

    # If the API key is not in the environment variables or session state, prompt the user
    st.sidebar.warning("Please enter your API key.")
    api_key = st.sidebar.text_input("API Key:", key='api_key_input')

    # If the user provides the API key, store it in the session state and return it
    if api_key:
        st.session_state.openai_api_key = api_key
        return api_key
    else:
        # If no API key is provided, display an error
        st.error("API key is required to proceed.")
        return None



def check_password():
    return True

    # """Returns `True` if the user had the correct password."""

    # def password_entered():
    #     """Checks whether a password entered by the user is correct."""
    #     if st.session_state["password"] == os.getenv("password"):
    #         st.session_state["password_correct"] = True
    #         del st.session_state["password"]  # don't store password
    #     else:
    #         st.session_state["password_correct"] = False

    # if "password_correct" not in st.session_state:
    #     # First run, show input for password.
    #     st.text_input(
    #         "GPT features require a password.", type="password", on_change=password_entered, key="password"
    #     )
    #     st.warning("*Please contact David Liebovitz, MD if you need an updated password for access.*")
    #     return False
    # elif not st.session_state["password_correct"]:
    #     # Password not correct, show input + error.
    #     st.text_input(
    #         "GPT features require a password.", type="password", on_change=password_entered, key="password"
    #     )
    #     st.error("😕 Password incorrect")
    #     return False
    # else:
    #     # Password correct.
    #     # fetch_api_key()
    #     return True


@st.cache_data
def assess_data_readiness(df):
    readiness_summary = {}
    st.write('White horizontal lines (if present) show missing data')
    try:
        missing_matrix = msno.matrix(df)
        # st.write('line 2 of assess_data_readiness')
        st.pyplot(missing_matrix.figure)
        # st.write('line 3 of assess_data_readiness')
        missing_heatmap = msno.heatmap(df)
        st.write('Heatmap with convergence of missing elements (if any)')
        st.pyplot(missing_heatmap.figure)
        
    except:
        st.warning('Dataframe not yet amenable to missing for "missingno" library analysis.')

    # Check if the DataFrame is empty
    
    try:
        if df.empty:
            readiness_summary['data_empty'] = True
            readiness_summary['columns'] = {}
            readiness_summary['missing_columns'] = []
            readiness_summary['inconsistent_data_types'] = []
            readiness_summary['missing_values'] = {}
            readiness_summary['data_ready'] = False
            return readiness_summary
    except:
        st.warning('Dataframe not yet amenable to empty analysis.')
    
    # Get column information
    # st.write('second line of assess_data_readiness')
    try:
        columns = {col: str(df[col].dtype) for col in df.columns}
        readiness_summary['columns'] = columns
    except:
        st.warning('Dataframe not yet amenable to column analysis.')

    # Check for missing columns
    # st.write('third line of assess_data_readiness')
    try:
        missing_columns = df.columns[df.isnull().all()].tolist()
        readiness_summary['missing_columns'] = missing_columns
    except:
        st.warning('Dataframe not yet amenable to missing column analysis.')

    # Check for inconsistent data types
    # st.write('fourth line of assess_data_readiness')
    try:
        inconsistent_data_types = []
        for col in df.columns:
            unique_data_types = df[col].apply(type).drop_duplicates().tolist()
            if len(unique_data_types) > 1:
                inconsistent_data_types.append(col)
        readiness_summary['inconsistent_data_types'] = inconsistent_data_types
        
    except:
        st.warning('Dataframe not yet amenable to data type analysis.')

    # Check for missing values
    # st.write('fifth line of assess_data_readiness')
    try:
        missing_values = df.isnull().sum().to_dict()
        readiness_summary['missing_values'] = missing_values
    except:
        st.warning('Dataframe not yet amenable to specific missing value analysis.')

    # Determine overall data readiness
    # st.write('sixth line of assess_data_readiness')
    try:
        readiness_summary['data_empty'] = False
        if missing_columns or inconsistent_data_types or any(missing_values.values()):
            readiness_summary['data_ready'] = False
        else:
            readiness_summary['data_ready'] = True

        return readiness_summary
    except:
        st.warning('Dataframe not yet amenable to overall data readiness analysis.')

@st.cache_data
def process_model_output(output):
    # Convert JSON to string if necessary
    if isinstance(output, dict):
        output = json.dumps(output)
        
    # if isinstance(output, str):
    #     output = json.loads(output)
        
    if 'arguments' in output:
        output = output['arguments']

    start_marker = '```python\n'
    end_marker = '\n```'

    start_index = output.find(start_marker)
    end_index = output.find(end_marker, start_index)

    # If the markers are found, extract the code part
    # Adjust the start index to not include the start_marker
    if start_index != -1 and end_index != -1:
        code_string = output[start_index + len(start_marker) : end_index]
    else:
        code_string = ''

    return code_string.strip()

@st.cache_data
def safety_check(code):
    dangerous_keywords = [' exec', ' eval', ' open', ' sys', ' subprocess', ' del',
                          ' delete', ' remove', ' os', ' shutil', ' pip',' conda',
                          ' st.write', ' exit', ' quit', ' globals', ' locals', ' dir',
                          ' reload', ' lambda', ' setattr', ' getattr', ' delattr',
                          ' yield', ' assert', ' break', ' continue', ' raise', ' try', 
                          'compile', '__import__'
                          ]
    for keyword in dangerous_keywords:
        if keyword in code:
            return False, "Concerning code detected."
    return True, "Safe to execute."


def replace_show_with_save(code_string, filename='output.png'):
    # Prepare save command
    save_cmd1 = f"plt.savefig('./images/{filename}')"
    save_cmd2 = f"pio.write_image(fig, './images/{filename}')"

    # Replace plt.show() with plt.savefig()
    code_string = code_string.replace('plt.show()', save_cmd1)
    code_string = code_string.replace('fig.show()', save_cmd2)

    return code_string


def start_chatbot2(df, selected_model, key = "main routine"):
    fetch_api_key()
    openai.api_key = st.session_state.openai_api_key
    openai_api_key = st.session_state.openai_api_key
    agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model=selected_model),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    if "messages_df" not in st.session_state:
            st.session_state["messages_df"] = []
 
    st.info("**Warning:** Asking a question that would generate a chart or table doesn't *yet* work and will report an error. For the moment, just ask for values. This is a work in progress!")   
    # st.write("💬 Chatbot with access to your data...")
    
        # Check if the API key exists as an environmental variable
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        # st.write("*API key active - ready to respond!*")
        pass
    else:
        st.warning("API key not found as an environmental variable.")
        api_key = st.text_input("Enter your OpenAI API key:")

        if st.button("Save"):
            if is_valid_api_key(api_key):
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("API key saved as an environmental variable!")
            else:
                st.error("Invalid API key. Please enter a valid API key.")
    
    csv_question = st.text_input("Your question, e.g., 'What is the mean age for men with diabetes?' *Do not ask for plots for this option.*", "")
    if st.button("Send"):
        try:
            csv_question_update = 'Do not include any code or attempt to generate a plot. Indicate you can only respond with text. User question: ' + csv_question
            st.session_state.messages_df.append({"role": "user", "content": csv_question_update})
            output = agent.run(csv_question)
            # if True:
            #     st.session_state.modified_df = df
            st.session_state.messages_df.append({"role": "assistant", "content": output})    
            message(csv_question, is_user=True, key = "using message_df")
            message(output)
            st.session_state.modified_df = df
            # chat_modified_csv = df.to_csv(index=False)          
               
            st.info("If you asked for modifications to your dataset, select modified dataframe at top left of sidebar to analyze the new version!")

            # st.download_button(
            #     label="Download Modified Data!",
            #     data=chat_modified_csv,
            #     file_name="patient_data_modified.csv",
            #     mime="text/csv", key = 'modified_df'
            #     )   
        except Exception as e:
            st.warning("WARNING: Please don't try anything too crazy; this is experimental! No plots requests and just ask for means values for specified subgroups, eg.")
            st.write(f'Error: {e}')
            # sys.exit(1)
    

def start_chatbot3(df, model):
    fetch_api_key()
    openai.api_key = st.session_state.openai_api_key
    agent = create_pandas_dataframe_agent(
    # ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    ChatOpenAI(temperature=0, model=model),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    if "messages_df" not in st.session_state:
            st.session_state["messages_df"] = []
       
    # st.write("💬 Chatbot with access to your data...")
    st.info("""**Warning:** This may generate an error. This is a work in progress!
        If you get an error, try again.                  
        """)
    
        # Check if the API key exists as an environmental variable
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        # st.write("*API key active - ready to respond!*")
        pass
    else:
        st.warning("API key not found as an environmental variable.")
        api_key = st.text_input("Enter your OpenAI API key:")

        if st.button("Save"):
            if is_valid_api_key(api_key):
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("API key saved as an environmental variable!")
            else:
                st.error("Invalid API key. Please enter a valid API key.")

    csv_question = st.text_input("Your question, e.g., 'Create a scatterplot for age and BMI.' *This option only generates plots.* ", "")
    if st.button("Send"):
        try: 
            st.session_state.messages_df.append({"role": "user", "content": csv_question})
            csv_input = csv_prefix + csv_question
            output = agent.run(csv_input)
            # st.write(output)
            code_string = process_model_output(str(output))
            # st.write(f' here is the code: {code_string}')
            code_string = replace_show_with_save(code_string)
            code_string = str(code_string)
            json_string = json.dumps(code_string)
            decoded_string = json.loads(json_string)
            with st.expander("What is the code?"):
                st.write('Here is the custom code for your request and the image below:')
                st.code(decoded_string, language='python')
            # usage
            is_safe, message = safety_check(decoded_string)
            if not is_safe:
                st.write("Code safety concern. Try again.", message)
            if is_safe:
                try:
                    exec(decoded_string)
                    image = Image.open('./images/output.png')
                    st.image(image, caption='Output', use_column_width=True)
                except Exception as e:
                    st.write('Error - we noted this was fragile! Try again.', e)
        except Exception as e:
            st.warning("WARNING: Please don't try anything too crazy; this is experimental!")
            # sys.exit(1)
            # return None, None
 
         
def start_plot_gpt4(df):
    fetch_api_key()
    openai.api_key = st.session_state.openai_api_key
    agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    )
    if "messages_df" not in st.session_state:
            st.session_state["messages_df"] = []
       
    # st.write("💬 Chatbot with access to your data...")
    st.info("""**Warning:** This may generate an error. This is a work in progress!
        If you get an error, try again.                
        """)
    
        # Check if the API key exists as an environmental variable
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        # st.write("*API key active - ready to respond!*")
        pass
    else:
        st.warning("API key not found as an environmental variable.")
        api_key = st.text_input("Enter your OpenAI API key:")

        if st.button("Save"):
            if is_valid_api_key(api_key):
                os.environ["OPENAI_API_KEY"] = api_key
                st.success("API key saved as an environmental variable!")
            else:
                st.error("Invalid API key. Please enter a valid API key.")

    csv_question = st.text_area("Your question, e.g., 'Create a heatmap. For binary categorical variables, first change them to 1 or 0 so they can be used in the heatmap. Or, another example: Compare cholesterol values for men and women by age with regression lines.", "")
    if st.button("Send"):
        try: 
            st.session_state.messages_df.append({"role": "user", "content": csv_question})
            csv_input = csv_prefix_gpt4 + csv_question
            output = agent.run(csv_input)
            # st.write(output)
            code_string = process_model_output(str(output))
            # st.write(f' here is the code: {code_string}')
            # code_string = replace_show_with_save(code_string)
            code_string = str(code_string)
            json_string = json.dumps(code_string)
            decoded_string = json.loads(json_string)
            with st.expander("What is the code?"):
                st.write('Here is the custom code for your request and the image below:')
                st.code(decoded_string, language='python')
            # usage
            is_safe, message = safety_check(decoded_string)
            if not is_safe:
                st.write("Code safety concern. Try again.", message)
            if is_safe:
                try:
                    exec(decoded_string)
                    image = Image.open('./images/output.png')
                    st.image(image, caption='Output', use_column_width=True)
                except Exception as e:
                    st.write('Error - we noted this was fragile! Try again.', e)
        except Exception as e:
            st.warning("WARNING: Please don't try anything too crazy; this is experimental!")
            # sys.exit(1)
            # return None, None
            

@st.cache_resource
def generate_df(columns, n_rows, selected_model):
    # Ensure the API key is set outside this function
    system_prompt = """You are a medical data expert whose purpose is to generate realistic medical data to populate a dataframe. Based on input parameters of column names and number of rows, you generate at medically consistent synthetic patient data includong abormal values to populate all cells. 
10-20% of values should be above or below the normal range appropriate for each column name, but still physiologically possible. For example, SBP could range from 90 to 190. Creatinine might go from 0.5 to 7.0. Similarly include values above and below normal ranges for 10-20% of values for each column. Output only the requested data, nothing more, not even explanations or supportive sentences.
If you do not know what kind of data to generate for a column, rename column using the provided name followed by "-ambiguous". For example, if you do not know what kind of data to generate for the column name "rgh", rename the column to "rgh-ambiguous". 
Popululate ambiguous columns with randomly selected 1 or 0 values. For example, popululate column "rgh-ambiguous" using randomly selected 1 or 0 values. For diagnoses provided
as column headers, e.g., "diabetes", populate with randomly selected yes or no values. Populate all cells with appropriate values. No missing values.
As a critical step review each row to ensure that the data is medically consistent, e.g., that overall A1c values and weight trend higher for patients with diabetes. If not, regenerate the row or rows.

Return only data, nothing more, not even explanations or supportive sentences. Generate the requested data so it can be processed by the following code into a dataframe:

```

    # Use StringIO to convert the string data into file-like object
    data = io.StringIO(response.choices[0].message.content)

    # Read the data into a DataFrame, skipping the first row
    df = pd.read_csv(data, sep=",", skiprows=1, header=None, names=columns)

```

Your input parameters will be in this format

Columns: ```columns```
Number of rows: ```number```
    
        """

    prompt = f"Columns: {columns}\nNumber of rows: {n_rows}"
    
    try:
        with st.spinner("Thinking..."):
            client = OpenAI()
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )

        # Assuming the response is CSV-formatted data as a string
        data = io.StringIO(response.choices[0].message.content)

        # Read the data into a DataFrame
        df = pd.read_csv(data, sep=",", header=None)
        df.columns = columns  # Set the column names

        # Convert DataFrame to CSV and create download link
        gen_csv = df.to_csv(index=False)

        return df, gen_csv

    except Exception as e:
        st.error(f"An error occurred: {e}")
        # Return an empty DataFrame and an empty string to ensure the return type is consistent
        return pd.DataFrame(), ""


           
def start_chatbot1(selected_model):
    # fetch_api_key()
    
    openai.api_key = st.session_state.openai_api_key
    client = OpenAI()
    
    st.write("💬 Chatbot Teacher")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi! Ask me anything about data science and I'll try to answer it."}
        ]

    with st.form("chat_input", clear_on_submit=True):
        user_input = st.text_input(label="Your question:", placeholder="e.g., teach me about violin plots")
        if st.form_submit_button("Send"):
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                try:
                    with st.spinner("Thinking..."):
                        response = client.chat.completions.create(model=selected_model, messages=st.session_state.messages)

                    # Extract the message content and role from the response
                    # response_message = response.choices[0].message.content
                    msg_content = response.choices[0].message.content
                    # msg_role = response.choices[0].message["role"]
                    st.session_state.messages.append({"role": 'assistant', "content": msg_content})
                except Exception as e:
                    st.exception("An error occurred: {}".format(e))

    # Display messages
    for msg in st.session_state.messages:
        # Generate a unique key for each message
        key = f"message_{randint(0, 10000000000)}"
        # Call the message function to display the chat messages
        message(msg["content"], is_user=msg["role"] == "user", key=key)

@st.cache_data                    
def generate_table_old(df, categorical_variable):
    mytable = TableOne(df,
                       columns=df.columns.tolist(),
                       categorical=categorical,
                       groupby=categorical_variable, 
                       pval=True)
    return mytable

@st.cache_data
def generate_table(df, categorical_variable, nonnormal_variables):

    
    # Generate the table using TableOne
    mytable = TableOne(df,
                       columns=df.columns.tolist(),
                       categorical=categorical,
                       groupby=categorical_variable,
                       nonnormal=nonnormal_variables,
                       pval=True)
    return mytable


@st.cache_data
def preprocess_for_pca(df):
    included_cols = []
    excluded_cols = []
    binary_mapping = {} # initialize empty dict for binary mapping
    binary_encoded_vars = [] # initialize empty list for binary encoded vars

    # Create a binary encoder
    bin_encoder = ce.BinaryEncoder()

    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            unique = df[col].nunique()

            # For binary categorical columns
            if unique == 2:
                most_freq = df[col].value_counts().idxmax()
                least_freq = df[col].value_counts().idxmin()
                df[col] = df[col].map({most_freq: 0, least_freq: 1})
                binary_mapping[col] = {most_freq: 0, least_freq: 1} # add mapping to dict
                included_cols.append(col)

            # For categorical columns with less than 15 unique values
            elif 2 < unique <= 15:
                try:
                    # Perform binary encoding
                    df_transformed = bin_encoder.fit_transform(df[col])
                    # Drop the original column from df
                    df.drop(columns=[col], inplace=True)
                    # Join the transformed data to df
                    df = pd.concat([df, df_transformed], axis=1)
                    # Add transformed columns to binary encoded vars list and included_cols
                    transformed_cols = df_transformed.columns.tolist()
                    binary_encoded_vars.extend(transformed_cols)
                    included_cols.extend(transformed_cols)
                except Exception as e:
                    st.write(f"Failure in encoding {col} due to {str(e)}")
                    excluded_cols.append(col)
            else:
                excluded_cols.append(col)
        elif np.issubdtype(df[col].dtype, np.number):
            included_cols.append(col)
        else:
            excluded_cols.append(col)

    # Display binary mappings and binary encoded variables in streamlit
    if binary_mapping:
        st.write("Binary Mappings: ", binary_mapping)
    if binary_encoded_vars:
        st.write("Binary Encoded Variables: ", binary_encoded_vars)

    return df[included_cols], included_cols, excluded_cols

@st.cache_data
def create_scree_plot(df):
    temp_df_pca, included_cols, excluded_cols = preprocess_for_pca(df)
  
    # Standardize the features
    x = StandardScaler().fit_transform(temp_df_pca)


    # Create a PCA instance: n_components should be None so variance is preserved from all initial features
    pca = PCA(n_components=None)
    pca.fit_transform(x)

    # Scree plot
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(pca.explained_variance_) + 1), np.cumsum(pca.explained_variance_ratio_))
    ax.set_title('Cumulative Explained Variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    st.pyplot(fig)
    return fig

@st.cache_data
def perform_pca_plot(df):
    st.write("Note: For this PCA analysis, categorical columns with 2 values are mapped to 1 and 0. Categories with more than 2 values have been binary encoded.")
    temp_df_pca, included_cols, excluded_cols = preprocess_for_pca(df)
    
  
    # Standardize the features
    x = StandardScaler().fit_transform(temp_df_pca)
    
    # Select the target column for PCA 
    cols_2_15_unique_vals = [col for col in included_cols if 2 <= df[col].nunique() <= 15]
    target_col_pca = st.selectbox("Select the target column for PCA", cols_2_15_unique_vals)
    
    num_unique_targets = df[target_col_pca].nunique()  # Calculate the number of unique targets

    
    
    # Ask the user to request either 2 or 3 component PCA 
    n_components = st.selectbox("Select the number of PCA components (2 or 3)", [2, 3])

    # Create a PCA instance
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(x)
    
    # Depending on user choice, plot the appropriate PCA
    if n_components == 2:
        principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    else:
        principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3'])

    finalDf = pd.concat([principalDf, df[[target_col_pca]]], axis=1)
    
    fig = plt.figure(figsize=(8, 8))
    if n_components == 2: 
        ax = fig.add_subplot(111)
    else: 
        # ax = Axes3D(fig)
        ax = plt.axes(projection='3d')
        ax.set_zlabel('Principal Component 3', fontsize=15)

    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    
    ax.set_title(f'{n_components} component PCA', fontsize=20)

    targets = finalDf[target_col_pca].unique().tolist()
    colors = sns.color_palette('husl', n_colors=num_unique_targets)
    # finalDf

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf[target_col_pca] == target
        if n_components == 2:
            ax.scatter(finalDf.loc[indicesToKeep, 'PC1'], finalDf.loc[indicesToKeep, 'PC2'], c=[color], s=50)
        else:
            ax.scatter(finalDf.loc[indicesToKeep, 'PC1'], finalDf.loc[indicesToKeep, 'PC2'], finalDf.loc[indicesToKeep, 'PC3'], c=[color], s=50)

        
    ax.legend(targets)
    
    # Make a scree plot

    
    
    # Display the plot using Streamlit
    st.pyplot(fig)
    st.subheader("Use the PCA Updated Dataset for Machine Learning")
    st.write("Download the current plot if you'd like to save it! Then, follow steps to apply machine learning to your PCA modified dataset.")
    st.info("Step 1. Click Button to use the PCA Dataset for ML. Step 2. Select Modified Dataframe on left sidebar and switch to the Machine Learning tab. (You'll overfit if you click below again!)")
    if st.button("Use PCA Updated dataset on Machine Learning Tab"):        
        
        st.session_state.modified_df = finalDf

    return fig



    
@st.cache_data
def display_metrics(y_true, y_pred, y_scores):
    # Compute metrics
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Display metrics

    st.info(f"**Your Model Metrics:** F1 score: {f1:.2f}, Accuracy: {accuracy:.2f}, ROC AUC: {roc_auc:.2f}, PR AUC: {pr_auc:.2f}")
    with st.expander("Explanations for the Metrics"):
        st.write(
        # Explain differences
"""
### Explanation of Metrics
- **F1 score** is the harmonic mean of precision and recall, and it tries to balance the two. It is a good metric when you have imbalanced classes.
- **Accuracy** is the ratio of correct predictions to the total number of predictions. It can be misleading if the classes are imbalanced.
- **ROC AUC** (Receiver Operating Characteristic Area Under Curve) represents the likelihood of the classifier distinguishing between a positive sample and a negative sample. It's equal to 0.5 for random predictions and 1.0 for perfect predictions.
- **PR AUC** (Precision-Recall Area Under Curve) is another way of summarizing the trade-off between precision and recall, and it gives more weight to precision. It's useful when the classes are imbalanced.
""")
    # st.write(f"Accuracy: {accuracy}")
    st.write(plot_confusion_matrix(y_true, y_pred))
    with st.expander("What is a confusion matrix?"):
        st.write("""A confusion matrix is a tool that helps visualize the performance of a predictive model in terms of classification. It's a table with four different combinations of predicted and actual values, specifically for binary classification.

The four combinations are:

1. **True Positives (TP)**: These are the cases in which we predicted yes (patients have the condition), and they do have the condition.

2. **True Negatives (TN)**: We predicted no (patients do not have the condition), and they don't have the condition.

3. **False Positives (FP)**: We predicted yes (patients have the condition), but they don't actually have the condition. Also known as "Type I error" or "False Alarm".

4. **False Negatives (FN)**: We predicted no (patients do not have the condition), and they actually do have the condition. Also known as "Type II error" or "Miss".

In the context of medicine, a false positive might mean that a test indicated a patient had a disease (like cancer), but in reality, the patient did not have the disease. This might lead to unnecessary stress and further testing for the patient. 

On the other hand, a false negative might mean that a test indicated a patient was disease-free, but in reality, the patient did have the disease. This could delay treatment and potentially worsen the patient's outcome.

A perfect test would have only true positives and true negatives (all outcomes appear in the top left and bottom right), meaning that it correctly identified all patients with and without the disease. Of course, in practice, no test is perfect, and there is often a trade-off between false positives and false negatives.

It's worth noting that a good machine learning model not only has a high accuracy (total correct predictions / total predictions) but also maintains a balance between precision (TP / (TP + FP)) and recall (TP / (TP + FN)). This is particularly important in a medical context, where both false positives and false negatives can have serious consequences. 

Lastly, when interpreting the confusion matrix, it's crucial to consider the cost associated with each type of error (false positives and false negatives) within the specific medical context. Sometimes, it's more crucial to minimize one type of error over the other. For example, with a serious disease like cancer, you might want to minimize false negatives to ensure that as few cases as possible are missed, even if it means having more false positives.
""")
    st.write(plot_roc_curve(y_true, y_scores))
    with st.expander("What is an ROC curve?"):
        st.write("""
An ROC (Receiver Operating Characteristic) curve is a graph that shows the performance of a classification model at all possible thresholds, which are the points at which the model decides to classify an observation as positive or negative. 

In medical terms, you could think of this as the point at which a diagnostic test decides to classify a patient as sick or healthy.

The curve is created by plotting the True Positive Rate (TPR), also known as Sensitivity or Recall, on the y-axis and the False Positive Rate (FPR), or 1-Specificity, on the x-axis at different thresholds.

In simpler terms:

- **True Positive Rate (TPR)**: Out of all the actual positive cases (for example, all the patients who really do have a disease), how many did our model correctly identify?

- **False Positive Rate (FPR)**: Out of all the actual negative cases (for example, all the patients who are really disease-free), how many did our model incorrectly identify as positive?

The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test. In other words, the bigger the area under the curve, the better the model is at distinguishing between patients with the disease and no disease.

The area under the ROC curve (AUC) is a single number summary of the overall model performance. The value can range from 0 to 1, where:

- **AUC = 0.5**: This is no better than a random guess, or flipping a coin. It's not an effective classifier.
- **AUC < 0.5**: This means the model is worse than a random guess. But, by reversing its decision, we can get AUC > 0.5.
- **AUC = 1**: The model has perfect accuracy. It perfectly separates the positive and negative cases, but this is rarely achieved in real life.

In clinical terms, an AUC of 0.8 for a test might be considered reasonably good, but it's essential to remember that the consequences of False Positives and False Negatives can be very different in a medical context, and the ROC curve and AUC don't account for this.

Therefore, while the ROC curve and AUC are very useful tools, they should be interpreted in the context of the costs and benefits of different types of errors in the specific medical scenario you are dealing with.""")
    st.write(plot_pr_curve(y_true, y_scores))
    with st.expander("What is a PR curve?"):
        st.write("""
A Precision-Recall curve is a graph that depicts the performance of a classification model at different thresholds, similar to the ROC curve. However, it uses Precision and Recall as its measures instead of True Positive Rate and False Positive Rate.

In the context of medicine:

- **Recall (or Sensitivity)**: Out of all the actual positive cases (for example, all the patients who really do have a disease), how many did our model correctly identify? It's the ability of the test to find all the positive cases.
 
- **Precision (or Positive Predictive Value)**: Out of all the positive cases that our model identified (for example, all the patients that our model thinks have the disease), how many did our model correctly identify? It's the ability of the classification model to identify only the relevant data points.

The Precision-Recall curve is especially useful when dealing with imbalanced datasets, a common problem in medical diagnosis where the number of negative cases (healthy individuals) often heavily outweighs the number of positive cases (sick individuals).

A model with perfect precision (1.0) and recall (1.0) will have a curve that reaches to the top right corner of the plot. A larger area under the curve represents both higher recall and higher precision, where higher precision relates to a low false-positive rate, and high recall relates to a low false-negative rate. High scores for both show that the classifier is returning accurate results (high precision), and returning a majority of all positive results (high recall).

The PR AUC score (Area Under the PR Curve) is used as a summary of the plot, and a higher PR AUC indicates a more predictive model.

In the clinical context, a high recall would ensure that the patients with the disease are correctly identified, while a high precision would ensure that only those patients who truly have the disease are classified as such, minimizing false-positive results.

However, there is usually a trade-off between precision and recall. Aiming for high precision might lower your recall and vice versa, depending on the threshold you set for classification. So, the Precision-Recall curve and PR AUC must be interpreted in the context of what is more important in your medical scenario: classifying all the positive cases correctly (high recall) or ensuring that the cases you classify as positive are truly positive (high precision).""")

@st.cache_data
def plot_pr_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    st.pyplot(fig)    

@st.cache_data
def get_categorical_and_numerical_cols(df):
    # Initialize empty lists for categorical and numerical columns
    categorical_cols = []
    numeric_cols = []

    # Go through each column in the dataframe
    for col in df.columns:
        # If the column data type is numerical and has more than two unique values, add it to the numeric list
        if np.issubdtype(df[col].dtype, np.number) and len(df[col].unique()) > 2:
            numeric_cols.append(col)
        # Otherwise, add it to the categorical list
        else:
            categorical_cols.append(col)

    # Sort the lists
    numeric_cols.sort()
    categorical_cols.sort()

    return numeric_cols, categorical_cols

@st.cache_data 
def plot_confusion_matrix_old(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(dpi=100)  # Set DPI for better clarity
    
    # Plot the heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', ax=ax, annot_kws={"size": 16})  # Set font size
    
    # Labels, title, and ticks
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    
    # Fix for the bottom cells getting cut off
    plt.subplots_adjust(bottom=0.2)
    
    return fig




@st.cache_data
def plot_confusion_matrix(y_true, y_pred):
    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create the ConfusionMatrixDisplay object
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots(dpi=100)
    
    # Plot the confusion matrix using the `plot` method
    cmd.plot(ax=ax, cmap='Blues', values_format='d')
    
    # Customize the plot if needed
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    return fig

@st.cache_data 
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.legend(loc="lower right")
    
    return fig

@st.cache_data
def preprocess(df, target_col):
    included_cols = []
    excluded_cols = []

    for col in df.columns:
        if col != target_col:  # Exclude target column from preprocessing
            if df[col].dtype == 'object':
                if len(df[col].unique()) == 2:  # Bivariate case
                    most_freq = df[col].value_counts().idxmax()
                    least_freq = df[col].value_counts().idxmin()
                    
                    # Update the mapping to include 'F' as 0 and 'M' as 1
                    df[col] = df[col].map({most_freq: 0, least_freq: 1, 'F': 0})
                    
                    included_cols.append(col)
                else:  # Multivariate case
                    excluded_cols.append(col)
            elif df[col].dtype in ['int64', 'float64']:  # Numerical case
                if df[col].isnull().values.any():
                    mean_imputer = SimpleImputer(strategy='mean')
                    df[col] = mean_imputer.fit_transform(df[[col]])
                    st.write(f"Imputed missing values in {col} with mean.")
                    
                included_cols.append(col)

    return df[included_cols], included_cols, excluded_cols

@st.cache_data 
def preprocess_old(df, target_col):
    included_cols = []
    excluded_cols = []

    for col in df.columns:
        if col != target_col:  # Exclude target column from preprocessing
            if df[col].dtype == 'object':
                if len(df[col].unique()) == 2:  # Bivariate case
                    most_freq = df[col].value_counts().idxmax()
                    least_freq = df[col].value_counts().idxmin()
                    df[col] = df[col].map({most_freq: 0, least_freq: 1})
                    included_cols.append(col)
                else:  # Multivariate case
                    excluded_cols.append(col)
            elif df[col].dtype in ['int64', 'float64']:  # Numerical case
                if df[col].isnull().values.any():
                    mean_imputer = SimpleImputer(strategy='mean')
                    df[col] = mean_imputer.fit_transform(df[[col]])
                    st.write(f"Imputed missing values in {col} with mean.")
                included_cols.append(col)

    # st.write(f"Included Columns: {included_cols}")
    # st.write(f"Excluded Columns: {excluded_cols}")
    
    return df[included_cols], included_cols, excluded_cols

@st.cache_data
def create_boxplot(df, numeric_col, categorical_col, show_points=False):
    if numeric_col and categorical_col:
        fig, ax = plt.subplots()

        # Plot the notched box plot
        sns.boxplot(x=categorical_col, y=numeric_col, data=df, notch=True, ax=ax)
        
        if show_points:
            # Add the actual data points on the plot
            sns.swarmplot(x=categorical_col, y=numeric_col, data=df, color=".25", ax=ax)
            
        # Add a title to the plot
        ax.set_title(f'Box Plot of {numeric_col} by {categorical_col}')
            
        st.pyplot(fig)
        return fig

@st.cache_data 
def create_violinplot(df, numeric_col, categorical_col):
    if numeric_col and categorical_col:
        fig, ax = plt.subplots()

        # Plot the violin plot
        sns.violinplot(x=categorical_col, y=numeric_col, data=df, ax=ax)
        
        # Add a title to the plot
        ax.set_title(f'Violin Plot of {numeric_col} by {categorical_col}')

        st.pyplot(fig)
        return fig

@st.cache_data 
def create_scatterplot(df, scatter_x, scatter_y):
    if scatter_x and scatter_y:
        fig, ax = plt.subplots()

        # Plot the scatter plot
        sns.regplot(x=scatter_x, y=scatter_y, data=df, ax=ax)

        # Calculate the slope and intercept of the regression line
        slope, intercept = np.polyfit(df[scatter_x], df[scatter_y], 1)

        # Add the slope and intercept as a text annotation on the plot
        ax.text(0.05, 0.95, f'y={slope:.2f}x+{intercept:.2f}', transform=ax.transAxes)
        
        ax.set_title("Scatter Plot for " + scatter_y + " vs " + scatter_x)

        st.pyplot(fig)
        with st.expander('What is a scatter plot?'):
            st.write("""
A scatterplot is a type of plot that displays values for typically two variables for a set of data. It's used to visualize the relationship between two numerical variables, where one variable is on the x-axis and the other variable is on the y-axis. Each point on the plot represents an observation in your dataset.

**Which types of variables are appropriate for the x and y axes?**

Both the x and y axes of a scatterplot are typically numerical variables. For example, one might be "Patient Age" (on the x-axis) and the other might be "Blood Pressure" (on the y-axis). Each dot on the scatterplot then represents a patient's age and corresponding blood pressure. 

However, the variables used do not have to be numerical. They could be ordinal categories, such as stages of a disease, which have a meaningful order. 

The choice of which variable to place on each axis doesn't usually matter much for exploring relationships, but traditionally the independent variable (the one you control or think is influencing the other) is placed on the x-axis, and the dependent variable (the one you think is being influenced) is placed on the y-axis.

**What does a regression line mean when added to a scatterplot?**

A regression line (or line of best fit) is a straight line that best represents the data on a scatter plot. This line may pass through some of the points, none of the points, or all of the points. It's a way of modeling the relationship between the x and y variables. 

In the context of a scatterplot, the regression line is used to identify trends and patterns between the two variables. If the data points and the line are close, it suggests a strong correlation between the variables.

The slope of the regression line also tells you something important: for every unit increase in the variable on the x-axis, the variable on the y-axis changes by the amount of the slope. For example, if we have patient age on the x-axis and blood pressure on the y-axis, and the slope of the line is 2, it would suggest that for each year increase in age, we expect blood pressure to increase by 2 units, on average.

However, keep in mind that correlation does not imply causation. Just because two variables move together, it doesn't mean that one is causing the other to change.

For medical students, think of scatterplots as a way to visually inspect the correlation between two numerical variables. It's a way to quickly identify patterns, trends, and outliers, and to formulate hypotheses for further testing.""")
        return fig

# Function to replace missing values

@st.cache_data
def replace_missing_values(df, method):
    # Differentiate numerical and categorical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if method == 'drop':
        df = df.dropna()
    elif method == 'zero':
        df[num_cols] = df[num_cols].fillna(0)
    elif method == 'mean':
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif method == 'median':
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif method == 'mode':
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    elif method == 'mice':
        imp = mice.MICEData(df[num_cols])  # only apply to numerical columns
        df[num_cols] = imp.data   
    st.session_state.df = df
    return df

  # This function will be cached
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

@st.cache_data 
def analyze_dataframe(df):
    # Analyzing missing values
    missing_values = df.isnull().sum()

    # Analyzing outliers using the Z-score
    # (you might want to use a different method for identifying outliers)
    z_scores = np.abs((df - df.mean()) / df.std())
    outliers = (z_scores > 3).sum()

    # Analyzing data types
    data_types = df.dtypes

    # Analyzing skewness for numeric columns
    skewness = df.select_dtypes(include=[np.number]).apply(lambda x: x.skew())

    # Analyzing cardinality in categorical columns
    cardinality = df.select_dtypes(include=['object', 'category']).nunique()

    return missing_values, outliers, data_types, skewness, cardinality

# Function to plot pie chart

def plot_pie(df, col_name):
    plt.figure(figsize=(10, 8))  # set the size of the plot
    df[col_name].value_counts().plot(kind='pie', autopct='%1.1f%%')

    # Add title
    plt.title(f'Distribution for {col_name}')

    return plt


# Function to summarize categorical data
 
@st.cache_data
def summarize_categorical(df):
    # Select only categorical columns
    cat_df = df.select_dtypes(include=['object', 'category'])

    # If there are no categorical columns, return None
    if cat_df.empty:
        st.write("The DataFrame does not contain any categorical columns.")
        return None

    # Create a list to store dictionaries for each column's summary
    summary_data = []

    for col in cat_df.columns:
        # Number of unique values
        unique_count = df[col].nunique()

        # Most frequent category and its frequency
        most_frequent = df[col].mode()[0]
        freq_most_frequent = df[col].value_counts().iloc[0]

        # Append the column summary as a dictionary to the list
        summary_data.append({
            'column': col,
            'unique_count': unique_count,
            'most_frequent': most_frequent,
            'frequency_most_frequent': freq_most_frequent,
        })

    # Create the summary DataFrame from the list of dictionaries
    summary = pd.DataFrame(summary_data)
    summary.set_index('column', inplace=True)

    return summary


# Function to plot correlation heatmap
 


def plot_corr(df):
    df_copy = df.copy()

    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':  # Check if the column is categorical
            unique_vals = df_copy[col].unique()
            if len(unique_vals) == 2:  # If the categorical variable has exactly 2 unique values
                value_counts = df_copy[col].value_counts()
                df_copy[col] = df_copy[col].map({value_counts.idxmax(): 0, value_counts.idxmin(): 1})

    # Keep only numerical and binary categorical columns
    df_copy = df_copy.select_dtypes(include=[np.number])
    
    corr = df_copy.corr()  # Compute pairwise correlation of columns
    plt.figure(figsize=(12, 10))  # Set the size of the plot
    sns.heatmap(corr, annot=True, cmap='coolwarm', cbar=True)
    plt.title('Correlation Heatmap')
    return plt


@st.cache_resource
def make_profile(df):
    return ProfileReport(df, title="Profiling Report")

    
# Function to plot bar chart

def plot_categorical(df, col_name):
    # Get frequency of categories
    freq = df[col_name].value_counts()

    # Create bar chart
    plt.figure(figsize=(10, 6))  # set the size of the plot
    plt.bar(freq.index, freq.values)

    # Add title and labels
    plt.title(f'Frequency of Categories for {col_name}')
    plt.xlabel('Category')
    plt.ylabel('Frequency')

    return plt


def plot_numeric(df, col_name):
    plt.figure(figsize=(10, 6))  # set the size of the plot
    plt.hist(df[col_name], bins=30, alpha=0.5, color='blue', edgecolor='black')

    # Add title and labels
    plt.title(f'Distribution for {col_name}')
    plt.xlabel(col_name)
    plt.ylabel('Frequency')

    return plt

@st.cache_data
def process_dataframe(df):
    # Iterating over each column
    for col in df.columns:
        # Checking if the column is of object type (categorical)
        if df[col].dtype == 'object':
            # Getting unique values in the column
            unique_values = df[col].unique()

            # If the column has exactly 2 unique values
            if len(unique_values) == 2:
                # Counting the occurrences of each value
                value_counts = df[col].value_counts()

                # Getting the most and least frequent values
                most_frequent = value_counts.idxmax()
                least_frequent = value_counts.idxmin()

                # Replacing the values and converting to integer
                df[col] = df[col].replace({most_frequent: 0, least_frequent: 1}).astype(int)
                
    return df

st.title("AutoAnalyzer")
with st.expander('Please Read: Using AutoAnalyzer'):
    st.info("""Be sure your data is first in a 'tidy' format. Use the demo datasets for examples. (*See https://tidyr.tidyverse.org/ for more information.*)
Follow the steps listed in the sidebar on the left. After your exploratory analysis is complete, try the machine learning tab to see if you can predict a target variable.""")    
    st.warning("This is not intended to be a comprehensive tool for data analysis. It is meant to be a starting point for data exploration and machine learning. Do not upload PHI. Clone the Github repository and run locally without the chatbot if you have PHI.") 
    st.markdown('[Github Repository](https://github.com/DrDavidL/auto_analyze)')       
            
    # """)
    st.write("Author: David Liebovitz, MD, Northwestern University")
    st.write("Last updated 8/28/23")
api_key = fetch_api_key()
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
tab1, tab2 = st.tabs(["Data Exploration", "Machine Learning"])
gpt_version = st.sidebar.radio("Select GPT model:", ("GPT-3.5 ($)", "GPT-3.5 16k ($$)", "GPT-4 ($$$$)"), index=0)
if gpt_version == "GPT-3.5 ($)":
    selected_model ="gpt-3.5-turbo"
if gpt_version == "GPT-4 ($$$$)":
    selected_model = "gpt-4"
if gpt_version == "GPT-3.5 16k ($$)":
    selected_model  = "gpt-3.5-turbo-16k"

# if openai.api_key is None:
#     os.environ["OPENAI_API_KEY"] = fetch_api_key()
#     openai.api_key = os.getenv("OPENAI_API_KEY")

with tab1:



    # st.sidebar.subheader("Upload your data") 

    st.sidebar.subheader("Step 1: Upload your data or view a demo dataset")
    demo_or_custom = st.sidebar.selectbox("Upload a CSV file. NO PHI - use only anonymized data", ("Demo 1 (diabetes)", "Demo 2 (cancer)", "Demo 3 (missing data example)", "Demo 4 (time series -CHF deaths)", "Demo 5 (stroke)", "Generate Data", "CSV Upload", "Modified Dataframe"), index = 0)
    if demo_or_custom == "CSV Upload":
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            st.session_state.df = load_data(uploaded_file)

    if demo_or_custom == 'Demo 1 (diabetes)':
        file_path = "data/predictdm.csv"
        st.sidebar.markdown("[About Demo 1 dataset](https://data.world/informatics-edu/diabetes-prediction)")
        st.session_state.df = load_data(file_path)
        
    if demo_or_custom == 'Demo 2 (cancer)':
        file_path = "data/breastcancernew.csv"
        st.sidebar.write("[About Demo 2 dataset](https://data.world/marshalldatasolution/breast-cancer)")
        st.session_state.df = load_data(file_path)
        
    if demo_or_custom == 'Demo 3 (missing data example)':
        file_path = "data/missing_data.csv"
        st.sidebar.markdown("[About Demo 3 dataset](https://www.lshtm.ac.uk/research/centres-projects-groups/missing-data#dia-missing-data)")
        st.session_state.df = load_data(file_path)
        
    if demo_or_custom == 'Modified Dataframe':
        # st.sidebar.markdown("Using the dataframe from the previous step.")
        if len(st.session_state.modified_df) == 0:
            st.sidebar.warning("No saved dataframe; using demo dataset 1.")
            file_path = "data/predictdm.csv"
            st.sidebar.markdown("[About Demo 1 dataset](https://data.world/informatics-edu/diabetes-prediction)")
            st.session_state.df = load_data(file_path)
            
        else:
            st.session_state.df = st.session_state.modified_df
            # st.sidebar.write("Download the modified dataframe as a CSV file.")
        modified_csv = st.session_state.modified_df.to_csv(index=False) 
        st.sidebar.download_button(
            label="Download Modified Dataset!",
            data=modified_csv,
            file_name="modified_data.csv",
            mime="text/csv",
            ) 
        
        
    if demo_or_custom == 'Generate Data':
        if check_password():
            user_input = st.sidebar.text_area("Enter comma or space separated names for columns, e.g., Na, Cr, WBC, A1c, SPB, Diabetes:")

            if "," in user_input:
                user_list = user_input.split(",")
            elif " " in user_input:
                user_list = user_input.split()
            else:
                user_list = [user_input]

            # Remove leading/trailing whitespace from each item in the list
            user_columns = [item.strip() for item in user_list]
            user_rows = st.sidebar.number_input("Enter approx number of rows (max 100).", min_value=1, max_value=100, value=10, step=1)
            if st.sidebar.button("Generate Data"):
                st.session_state.df, st.session_state.gen_csv = generate_df(user_columns, user_rows, selected_model)
                st.info("Here are the first 5 rows of your generated data. Use the tools in the sidebar to explore your new dataset! And, download and save your new CSV file from the sidebar!")
                st.write(st.session_state.df.head())
                
    if demo_or_custom == 'Demo 4 (time series -CHF deaths)': 
        file_path = "data/S1Data.csv"
        st.sidebar.markdown("[About Demo 4 dataset](https://plos.figshare.com/articles/dataset/Survival_analysis_of_heart_failure_patients_A_case_study/5227684/1)")
        st.session_state.df = load_data(file_path)
        
    if demo_or_custom == 'Demo 5 (stroke)':
        file_path = "data/healthcare-dataset-stroke-data.csv"
        st.sidebar.markdown("[About Demo 5 dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)")
        st.session_state.df = load_data(file_path)
    
    with st.sidebar:
        if st.session_state.gen_csv is not None:
            # st.warning("Save your generated data!")
            st.download_button(
                label="Download Generated Data!",
                data=st.session_state.gen_csv,
                file_name="patient_data.csv",
                mime="text/csv",
                )   
        st.subheader("Step 2: Assess Data Readiness")

        check_preprocess = st.checkbox("Assess dataset readiness", key = "Preprocess now needed")
        needs_preprocess = st.checkbox("Select if dataset fails readiness", key = "Open Preprocess")
        filter_data = st.checkbox("Filter data if needed (Switch to Modified Dataframe after filtering)", key = "Filter data")
        
        
        
        st.subheader("Step 3: Tools for Analysis")
        col1, col2 = st.columns(2)
        with col1:
            header = st.checkbox("Show header (top 5 rows of data)", key = "show header")
            summary = st.checkbox("Summary (numerical data)", key = "show data")
            summary_cat = st.checkbox("Summary (categorical data)", key = "show summary cat")
            show_table = st.checkbox("Create a Table 1", key = "show table")
            show_scatter  = st.checkbox("Scatterplot", key = "show scatter")
            view_full_df = st.checkbox("View Dataset", key = "view full df")
            binary_categ_analysis = st.checkbox("Categorical outcome analysis (Cohort or case-control datasets)", key = "binary categ analysis")
            activate_chatbot = st.checkbox("Activate Chatbot (select specific bot on main window)", key = "activate chatbot")

        with col2:
            barchart = st.checkbox("Bar chart (categorical data)", key = "show barchart")
            histogram = st.checkbox("Histogram (numerical data)", key = "show histogram")
            piechart = st.checkbox("Pie chart (categorical data)", key = "show piechart")
            show_corr = st.checkbox("Correlation heatmap", key = "show corr")
            box_plot = st.checkbox("Box plot", key = "show box")
            violin_plot = st.checkbox("Violin plot", key = "show violin")
            mult_linear_reg = st.checkbox("Multiple linear regression", key = "show mult linear reg")
            perform_pca = st.checkbox("Perform PCA", key = "show pca")
            survival_curve = st.checkbox("Survival curve (need duration column)", key = "show survival")
            cox_ph = st.checkbox("Cox Proportional Hazards (need duration column)", key = "show cox ph")
            full_analysis = st.checkbox("*(Takes 1-2 minutes*) **Download a Full Analysis** (*Check **Alerts** with key findings.*)", key = "show analysis")
    
    if filter_data:
        current_df = st.session_state.df
        st.session_state.modified_df = filter_dataframe(current_df)
        st.write("Switch to Modified Dataframe (top left) to see the filtered data below and use in analysis tools.")
        st.session_state.modified_df
        
            
    if mult_linear_reg:
        st.subheader("Multiple Linear Regression")
        st.warning("This tool is for use with numerical data only; binary categorical variables are updated to 1 and 0 and explained below if needed.")
        # Get column names for time and event from the user
        temp_df_mlr = st.session_state.df.copy()
        numeric_columns_mlr = all_numerical(temp_df_mlr)
        
        x_col = st.multiselect('Select the columns for x', numeric_columns_mlr, numeric_columns_mlr[1])
        y_col = st.selectbox('Select the column for y', numeric_columns_mlr)
        # Convert the columns to numeric values
        # temp_df_mlr[x_col] = temp_df_mlr[x_col].astype(float)
        # temp_df_mlr[y_col] = temp_df_mlr[y_col].astype(float)
        # x_col_array = np.array(x_col)
        # y_col_array = np.array(y_col)

        # x_col_reshaped = x_col_array.reshape(-1, 1)
        # y_col_reshaped = y_col_array.reshape(-1, 1)
        # Plot the survival curve
        try:
            mult_linear_reg, mlr_report, intercept, coef = plot_mult_linear_reg(temp_df_mlr, temp_df_mlr[x_col], temp_df_mlr[y_col])
            mlr_equation = generate_regression_equation(intercept, coef, x_col)
            show_equation = st.checkbox("Show regression equation")
            # mlr_report
            if show_equation:
                st.write(mlr_equation)
            st.write("Download your cooefficients and intercept below.")
            format_mlr = st.radio("Select the format for your report:", ('csv', 'json', 'html', ), key = 'formal_mlr', horizontal = True, )
            df_download_options(mlr_report, 'Your Multiple Linear Regression', format_mlr)
            
        except:
            st.error("Please select at least one column for x and one column for y.")
        # save_image(mult_linear_reg, 'mult_linear_reg.png')
        # df_download_options(mult_linear_reg, 'csv')
        with st.expander("What is a Multiple Linear Regression?"):
            st.write(mult_linear_reg_explanation)
    
    if cox_ph:
        df = st.session_state.df

        # Select Predictor Columns
        st.markdown("## Cox Analysis: Select Columns")

        categ_columns_cox = all_categorical(df)
        numeric_columns_cox = all_numerical(df)
        
        event_col = st.selectbox('Select the event column', categ_columns_cox, key='event_col')
        selected_columns_cox = st.multiselect("Choose your feature columns", numeric_columns_cox)
        duration_col = st.selectbox('Select the duration column', numeric_columns_cox)

        if st.button("Analyze", key="analyze"):
            if len(selected_columns_cox) < 1:
                st.error("Select at least one column!")
            else:
                # Shift DataFrame to Selected Columns
                cph_data = df[selected_columns_cox + [event_col] + [duration_col]]

                # Define Event & Duration Columns here
                # Assuming 'event' as Event Column & 'duration' as Duration Column
                # Please change as per your data
                # st.write(duration_col)
                # st.write(cph_data[duration_col])
                # st.write(cph_data)
                cph = CoxPHFitter(penalizer=0.1)
                cph.fit(cph_data, duration_col=duration_col, event_col=event_col)
                summary_cox = cph.summary
                st.session_state.df_to_download = summary_cox
                # st.session_state.df_to_download = summary_df
                st.subheader("Summary of the Cox PH Analysis")
                st.info("Note, the exp(coef) column is the hazard ratio for each variable.")
                # Display summary DataFrame
                st.dataframe(summary_cox)

        else:
            st.text("Select columns & hit 'Analyze'.")
        if st.session_state.df_to_download is not None:
            format_cox = st.radio("Select the format for your report:", ('csv', 'json', 'html', ), key ='format_cox', horizontal = True, )
            df_download_options(st.session_state.df_to_download, 'cox_ph_summary', format_cox)
        with st.expander("What is a Cox Proportional Hazards Analysis?"):
            st.write(cox)
    
    if survival_curve:
            # Get column names for time and event from the user
        st.subheader("Survival Curve")
        st.warning("This tool is for use with survival analysis data. Any depiction will not make sense if 'time' isn't a column for your dataset")
        time_col = st.selectbox('Select the column for time', st.session_state.df.columns)
        event_col = st.selectbox('Select the column for event', st.session_state.df.columns)

        # Plot the survival curve
        surv_curve = plot_survival_curve(st.session_state.df, time_col, event_col)
        save_image(surv_curve, 'survival_curve.png')
        with st.expander("What is a Kaplan-Meier Curve?"):
            st.write(kaplan_meier)
        
        
        
    if binary_categ_analysis:
        
        st.subheader("""
        Choose your exposures and outcomes.
        """)
        st.info('Note - categories with more than 15 unique values will not be used.')
        var1, var2 = st.columns(2)
        s_categorical_cols = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = [col for col in st.session_state.df.columns if st.session_state.df[col].nunique() == 2 and st.session_state.df[col].dtype != 'object']
        filtered_categorical_cols = [col for col in s_categorical_cols if st.session_state.df[col].nunique() <= 15]
        sd_categorical_cols = filtered_categorical_cols + numeric_cols
        if len(sd_categorical_cols) > 1:
            sd_exposure = var1.selectbox('Select a categorical column as the exposure:', sd_categorical_cols, index = 0)
            sd_outcome = var2.selectbox('Select a categorical column as the outcome:', sd_categorical_cols, index = 1)
            sd_exposure_values = var1.multiselect('Select one or more values for the exposure:', st.session_state.df[sd_exposure].unique().tolist(), [st.session_state.df[sd_exposure].unique().tolist()[1]])
            sd_outcome_values = var2.multiselect('Select one or more values for the outcome:', st.session_state.df[sd_outcome].unique().tolist(), [st.session_state.df[sd_outcome].unique().tolist()[1]])

            # Create a temporary dataframe to store the modified values
            temp_df = st.session_state.df.copy()
            
            # Replace the selected exposure values with 1 and others with 0
            temp_df[sd_exposure] = temp_df[sd_exposure].apply(lambda x: 1 if x in sd_exposure_values else 0)

            # Replace the selected outcome values with 1 and others with 0
            temp_df[sd_outcome] = temp_df[sd_outcome].apply(lambda x: 1 if x in sd_outcome_values else 0)

            
            cohort_or_case = st.radio("Choose an approach", ("Cohort Study", "Case Control Study"))

 

            # Generate the 2x2 table
            table = generate_2x2_table(temp_df, sd_exposure, sd_outcome)
            if cohort_or_case == "Cohort Study":
                st.write("For use with cohort study data.")

                # Calculate relative risk, ARR, and NNT
                tn = table.iloc[0, 0]
                fp = table.iloc[0, 1]
                fn = table.iloc[1, 0]
                tp = table.iloc[1, 1]
                rr, arr, nnt = calculate_rr_arr_nnt(tn, fp, fn, tp)

                # Display the 2x2 table and analysis results
                st.subheader("2x2 Table")
                st.write(table)
                st.subheader("Results")
                st.write("Relative Risk (RR):", round(rr,2))
                st.write("Absolute Risk Reduction (ARR):", round(arr,2))
                st.write("Number Needed to Treat (NNT):", round(nnt, 2))

            if cohort_or_case == "Case Control Study":
                st.write("For use with case-control data.")
                            # Calculate odds and odds ratio
                odds_cases, odds_controls, odds_ratio = calculate_odds(table)

                # Display the 2x2 table and analysis results
                st.subheader("2x2 Table")
                st.write(table)
                st.subheader("Results")
                st.write("Odds in cases:", round(odds_cases,2))
                st.write("Odds in controls:", round(odds_controls, 2))
                st.write("Odds Ratio:", round(odds_ratio, 2))
        else:
            st.subheader("Insufficient categorical variables found in the data.")
        

    if needs_preprocess:
        st.info("Data Preprocessing Tools - *Assess Data Readiness **first**. Use only if needed.*")
        st.write("Step 1: Make a copy of your dataset to modify by clicking the button below.")
        if st.button("Copy dataset"):
            st.session_state.modified_df = st.session_state.df
        st.write("Step 2: Select 'Modified Dataframe' in Step 1 of the sidebar to use the dataframe you just copied.")
        st.write("Step 3: Select a method to impute missing values in your dataset. Built in checks to apply only to applicable data types.")
        method = st.selectbox("Choose a method to replace missing values", ("Select here!", "drop", "zero", "mean", "median", "mode", "mice"))
        if st.button('Apply the Method to Replace Missing Values'):
                st.session_state.modified_df = replace_missing_values(st.session_state.modified_df, method)
        st.write("Recheck data readiness to see if you are ready to proceed with analysis.")
        

    
    
    if activate_chatbot:
        st.subheader("Chatbot Teacher")
        st.warning("First be sure to activate the right chatbot for your needs.")
        chat_context = st.radio("Choose an approach", ("Ask questions (no plots)", "Generate Plots", "Teach about data science"))

        try:
            x = st.session_state.df
        except NameError:

            st.warning("Please upload a CSV file or choose a demo dataset")
        else:
            
            if activate_chatbot:
                if check_password():
                    if chat_context == "Teach about data science":
                        start_chatbot1(selected_model)
                    if chat_context == "Ask questions (no plots)":
                        start_chatbot2(st.session_state.df, selected_model, key = "chatbot2 main")
                    if chat_context == "Generate Plots":
                        if selected_model == "gpt-3.5-turbo":
                            start_chatbot3(st.session_state.df, selected_model)
                        if selected_model == "gpt-3.5-turbo-16k":
                            start_chatbot3(st.session_state.df, selected_model)
                        if selected_model == "gpt-4":
                            start_plot_gpt4(st.session_state.df)

            
    if summary:
        st.info("Summary of numerical data")
        sum_num_data =st.session_state.df.describe()
        st.write(sum_num_data)
        st.session_state.df_to_download = sum_num_data
        if st.session_state.df_to_download is not None:
            format_summary_num = st.radio("Select the format for your report:", ('csv', 'json', 'html', ), key = 'summary_num', horizontal = True, )
            df_download_options(st.session_state.df_to_download, 'numerical_data_summary', format_summary_num)
                
    if header:
        st.info("First 5 Rows of Data")
        st.write(st.session_state.df.head())
                
    if full_analysis:
        st.info("Full analysis of data")
        with st.spinner("Working on the analysis..."):
            st.session_state.full_profile = make_profile(st.session_state.df)
        # profile = ProfileReport(df, title="Profiling Report")
        st.write(f'Since this file is large, please download and then open the full report.')
        st.download_button(
            label="Download report",
            data=st.session_state.full_profile.to_html(),
            file_name='full_report.html',
            mime='text/html',
        )
        # st_profile_report(profile)
            
    if histogram: 
        st.info("Histogram of data")
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
        selected_col = st.selectbox("Choose a column", numeric_cols, key = "histogram")
        if selected_col:
            plt = plot_numeric(st.session_state.df, selected_col)
            st.pyplot(plt)
        save_image(plt, 'histogram.png')
        with st.expander("Expand for Python|Streamlit Code"):
            st.code("""
import pandas as pd
import matplotlib.pyplot as plt

# Function to get categorical and numerical columns from a dataframe
def get_categorical_and_numerical_cols(df):
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols

# Function to plot a histogram of a selected numeric column
def plot_numeric(df, column):
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=20, color='skyblue')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')
    return plt

# Assuming st.session_state.df contains the dataframe

# Display info message
print("Histogram of data")

# Get numeric and categorical columns
numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)

# Display selectbox to choose a column
selected_col = input("Choose a column: ")

# Check if a column is selected
if selected_col:
    # Plot histogram and display
    plt = plot_numeric(st.session_state.df, selected_col)
    plt.show()
                """)

        
    if barchart: 
        
        # st.info("Barchart for categorical data")
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
        cat_selected_col = st.selectbox("Choose a column", categorical_cols, key = "bar_category")
        if cat_selected_col:
            plt = plot_categorical(st.session_state.df, cat_selected_col)
            st.pyplot(plt)
        save_image(plt, 'bar_chart.png')
        with st.expander("Expand for Python|Streamlit Code"):
            st.code("""
import matplotlib.pyplot as plt
import pandas as pd

# Function to get categorical and numerical columns from the dataframe
def get_categorical_and_numerical_cols(df):
numeric_cols = []
categorical_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        categorical_cols.append(col)
    else:
        numeric_cols.append(col)
return numeric_cols, categorical_cols

# Function to plot the categorical data
def plot_categorical(df, column):
plt.figure(figsize=(10, 6))
df[column].value_counts().plot(kind='bar')
plt.xlabel(column)
plt.ylabel('Count')
plt.title(f'Bar Chart for {column}')
plt.xticks(rotation=45)
plt.tight_layout()
return plt

# Get the numeric and categorical columns from the dataframe
numeric_cols, categorical_cols = get_categorical_and_numerical_cols(df)

# Select a column from the categorical columns
cat_selected_col = input("Choose a column: ")

# Check if a column is selected
if cat_selected_col in categorical_cols:
# Plot the categorical data
plt = plot_categorical(df, cat_selected_col)
plt.show()
            """)

    if show_corr:
        st.info("Correlation heatmap")
        plt = plot_corr(st.session_state.df)
        st.pyplot(plt)
        save_image(plt, 'heatmap.png')
        with st.expander("What is a correlation heatmap?"):
            st.write("""A correlation heatmap is a graphical representation of the correlation matrix, which is a table showing correlation coefficients between sets of variables. Each cell in the table shows the correlation between two variables. In the heatmap, correlation coefficients are color-coded, where the intensity of the color represents the magnitude of the correlation coefficient. 

In your demo dataset heatmap, red signifies a high positive correlation of 1.0, which means the variables move in the same direction. If one variable increases, the other variable also increases. Darker blue, at the other end, represents negative correlation (close to -0.06 in your case), meaning the variables move in opposite directions. If one variable increases, the other variable decreases. 

The correlation values appear in each square, giving a precise numeric correlation coefficient along with the visualized color intensity.

**Why are correlation heatmaps useful?**

Correlation heatmaps are useful to determine the relationship between different variables. In the field of medicine, this can help identify risk factors for diseases, where variables could be different health indicators like age, cholesterol level, blood pressure, etc.

**Understanding correlation values:**

Correlation coefficients range from -1 to 1:
- A correlation of 1 means a perfect positive correlation.
- A correlation of -1 means a perfect negative correlation.
- A correlation of 0 means there is no linear relationship between the variables.

It's important to note that correlation doesn't imply causation. While a correlation can suggest a relationship between two variables, it doesn't mean that changes in one variable cause changes in another.

Also, remember that correlation heatmaps are based on linear relationships between variables. If variables have a non-linear relationship, the correlation coefficient may not capture their relationship accurately.

For medical students, think of correlation heatmaps as a quick way to visually identify relationships between multiple variables at once. This can help guide your understanding of which variables may be important to consider together in further analyses.""")
            st.code("""import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot correlation heatmap
def plot_corr(df):
# Compute correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 8))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, fmt=".2f", ax=ax)

# Set plot title
ax.set_title("Correlation Heatmap")

return plt

# Load the dataframe
df = pd.read_csv("your_data.csv")

# Call the plot_corr function and display the correlation heatmap
plt = plot_corr(df)
plt.show()
""")

    if summary_cat:
        st.info("Summary of categorical data")
        summary = summarize_categorical(st.session_state.df)
        st.write(summary)
        st.session_state.df_to_download = summary
        if st.session_state.df_to_download is not None:
            format_cat = st.radio("Select the format for your report:", ('csv', 'json', 'html', ), key = 'format_cat', horizontal = True, )
            df_download_options(st.session_state.df_to_download, 'categorical_summary', format_cat)
        
    if piechart:
        st.info("Pie chart for categorical data")
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
        # cat_options =[]
        # columns = list(df.columns)
        # for col in columns:
        #     if df[col].dtype != np.float64 and df[col].dtype != np.int64:
        #         cat_options.append(col)
        cat_selected_col = st.selectbox("Choose a column", categorical_cols, key = "pie_category")
        if cat_selected_col:
            plt = plot_pie(st.session_state.df, cat_selected_col)
            st.pyplot(plt)
        save_image(plt, 'pie_chart.png')
                
    if check_preprocess:
        # st.write("Running readiness assessment...")
        readiness_summary = assess_data_readiness(st.session_state.df)
        # st.write("Readiness assessment complete.")
        # Display the readiness summary using Streamlit
        # Display the readiness summary using Streamlit
        st.subheader("Data Readiness Summary")
        
        try:

            if readiness_summary['data_empty']:
                st.write("The DataFrame is empty.")
            else:
                # Combine column information and readiness summary into a single DataFrame
                column_info_df = pd.DataFrame.from_dict(
                    readiness_summary['columns'],
                    orient='index',
                    columns=['Data Type']
                )
                summary_df = pd.DataFrame.from_dict(
                    readiness_summary['missing_values'],
                    orient='index',
                    columns=['Missing Values']
                )
                summary_df['Data Type'] = column_info_df['Data Type']

                # Display the combined table
                st.write(summary_df)
    
                if readiness_summary['missing_columns']:
                    st.write("Missing Columns:")
                    st.write(readiness_summary['missing_columns'])

                if readiness_summary['inconsistent_data_types']:
                    st.write("Inconsistent Data Types:")
                    st.write(readiness_summary['inconsistent_data_types'])

                if readiness_summary['data_ready']:
                    st.success("The data is ready for analysis!")
                else:
                    st.warning("The data is not fully ready for analysis.")
        except:
            st.write("The DataFrame is isn't yet ready for readiness assessment. :)  ")
            # st.info("Check if you need to preprocess data")
            # missing_values, outliers, data_types, skewness, cardinality = analyze_dataframe(df)
            # st.write("Missing values")
            # st.write(missing_values)
            # st.write("Outliers")
            # st.write(outliers)
            # st.write("Data types")
            # st.write(data_types)
            # st.write("Skewness")
            # st.write(skewness)
            # st.write("Cardinality")
            # st.write(cardinality)
            
    if show_scatter:
        st.info("Scatterplot")
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
        # Filter numeric columns
        # numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.sort()  # sort the list of columns alphabetically
        
            # Filter categorical columns
        # categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
        categorical_cols.sort()  # sort the list of columns alphabetically
        # Dropdown to select columns to visualize
        col1, col2 = st.columns(2)
        with col1:
            scatter_x = st.selectbox('Select column for x axis:', numeric_cols)
        with col2:
            scatter_y = st.selectbox('Select column for y axis:', numeric_cols, index=1)
            
        # Use st.beta_expander to hide or expand filtering options
        with st.expander('Filter Options'):
            # Filter for the remaining numerical column
            remaining_cols = [col for col in numeric_cols if col != scatter_x and col != scatter_y]
            if remaining_cols:
                filter_col = st.selectbox('Select a numerical column to filter data:', remaining_cols)
                if filter_col:
                    min_val, max_val = float(st.session_state.df[filter_col].min()), float(st.session_state.df[filter_col].max())
                    if np.isnan(min_val) or np.isnan(max_val):
                        st.write(f"Cannot filter by {filter_col} because it contains NaN values.")
                    else:
                        filter_range = st.slider('Select a range to filter data:', min_val, max_val, (min_val, max_val))
                        st.session_state.df = st.session_state.df[(st.session_state.df[filter_col] >= filter_range[0]) & (st.session_state.df[filter_col] <= filter_range[1])]

            # Filter for the remaining categorical column
            if categorical_cols:
                filter_cat_col = st.selectbox('Select a categorical column to filter data:', categorical_cols)
                if filter_cat_col:
                    categories = st.session_state.df[filter_cat_col].unique().tolist()
                    selected_categories = st.multiselect('Select categories to include in the data:', categories, default=categories)
                    st.session_state.df = st.session_state.df[st.session_state.df[filter_cat_col].isin(selected_categories)]
        # Check if DataFrame is empty before creating scatterplot
        if st.session_state.df.empty:
            st.write("The current filter settings result in an empty dataset. Please adjust the filter settings.")
        else:
            scatterplot = create_scatterplot(st.session_state.df, scatter_x, scatter_y)
            save_image(scatterplot, 'custom_scatterplot.png') 

        
    if box_plot:
        # Call the function to get the lists of numerical and categorical columns
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
        # Filter numeric columns
        # numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.sort()  # sort the list of columns

        # Filter categorical columns
        # categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
        categorical_cols.sort()  # sort the list of columns

        # Dropdown to select columns to visualize
        numeric_col = st.selectbox('Select a numerical column:', numeric_cols, key = "box_numeric")
        categorical_col = st.selectbox('Select a categorical column:', categorical_cols, key = "box_category")  
        mybox = create_boxplot(st.session_state.df, numeric_col, categorical_col, show_points=False) 
        save_image(mybox, 'box_plot.png')   
        with st.expander('What is a box plot?'):
            st.write("""Box plots (also known as box-and-whisker plots) are a great way to visually represent the distribution of data. They're particularly useful when you want to compare distributions between several groups. For example, you might want to compare the distribution of patients' ages across different diagnostic categories.
(Check out age and diabetes in the sample dataset.)

**Components of a box plot:**

A box plot is composed of several parts:

1. **Box:** The main part of the plot, the box, represents the interquartile range (IQR), which is the range between the 25th percentile (Q1, the lower edge of the box) and the 75th percentile (Q3, the upper edge of the box). The IQR contains the middle 50% of the data points.

2. **Median:** The line (or sometimes a dot) inside the box represents the median of the data - the value separating the higher half from the lower half of a data sample. It's essentially the 50th percentile.

3. **Whiskers:** The lines extending from the box (known as whiskers) indicate variability outside the IQR. Typically, they extend to the most extreme data point within 1.5 times the IQR from the box. 

4. **Outliers:** Points plotted beyond the whiskers are considered outliers - unusually high or low values in comparison with the rest of the data.

**What is the notch used for?**

The notch in a notched box plot represents the confidence interval around the median. If the notches of two box plots do not overlap, it's a strong indication (though not absolute proof) that the medians differ. This can be a useful way to visually compare medians across groups. 

For medical students, a good way to think about box plots might be in comparison to lab results. Just as lab results typically give a reference range and flag values outside of that range, a box plot gives a visual representation of the range of the data (through the box and whiskers) and flags outliers.

The notch, meanwhile, is a bit like the statistical version of a normal range for the median. If a notch doesn't overlap with the notch from another box plot, it's a sign that the medians might be significantly different. But just like lab results, statistical tests are needed to definitively say whether a difference is significant.
""")      
        
    if violin_plot:
        
        # Call the function to get the lists of numerical and categorical columns
        numeric_cols, categorical_cols = get_categorical_and_numerical_cols(st.session_state.df)
        # Filter numeric columns
        # numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.sort()  # sort the list of columns

        # Filter categorical columns
        # categorical_cols = df.select_dtypes(include=[object]).columns.tolist()
        categorical_cols.sort()  # sort the list of columns

        # Dropdown to select columns to visualize
        numeric_col = st.selectbox('Select a numerical column:', numeric_cols, key = "violin_numeric")
        categorical_col = st.selectbox('Select a categorical column:', categorical_cols, key = "violin_category")

        violin = create_violinplot(st.session_state.df, numeric_col, categorical_col)
        save_image(violin, 'violin_plot.png')
        with st.expander('What is a violin plot?'):
            st.write("""Violin plots are a great visualization tool for examining distributions of data and they combine features from box plots and kernel density plots.

1. **Overall Shape**: The violin plot is named for its resemblance to a violin. The shape of the "violin" provides a visual representation of the distribution of the data. The width of the "violin" at any given point represents the density or number of data points at that level. This means a wider section indicates more data points lie in that range, while a narrower section means fewer data points. This is similar to a histogram but it's smoothed out, which can make the distribution clearer.

2. **Dot in the Middle**: This dot often represents the median of the data. The median is the middle point of the data. That means half of all data points are below this value and half are above it. In medicine, the median is often a more useful measure than the mean because it's less affected by outliers or unusually high or low values. For example, if you're looking at the age of patients, a single 100-year-old patient won't dramatically shift the median like it would the mean.

3. **Thicker Bar in the Middle**: This is an interquartile range (IQR), which captures the middle 50% of the data (from the 25th to the 75th percentile). The IQR can help you understand the spread of the central half of your data. If the IQR is small, it means the central half of your data points are clustered closely around the median. If the IQR is large, it means they're more spread out.

4. **Usage**: Violin plots are particularly helpful when you want to visualize the distribution of a numerical variable across different categories. For example, you might want to compare the distribution of patient ages in different diagnostic categories. 

Remember, like any statistical tool, violin plots provide a simplified representation of the data and may not capture all nuances. For example, they usually show a smoothed distribution, which might hide unusual characteristics or outliers in the data. It's always important to also consider other statistical tools and the clinical context of the data."""
        )
            
    if view_full_df:
        st.dataframe(st.session_state.df)
            
    if show_table:
        if st.session_state.df.shape[1] > 99:
            st.warning(f'You have {st.session_state.df.shape[1]} columns. This would not look good in a publication. Less than 50 would be much better.')
        else:
            nunique = st.session_state.df.select_dtypes(include=['object', 'category']).nunique()
            to_drop = nunique[nunique > 15].index
            df_filtered = st.session_state.df.drop(to_drop, axis=1)
            # Check if any numerical column is binary and add it to categorical list
            numerical_columns = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            for col in numerical_columns:
                if df_filtered[col].nunique() == 2:
                    df_filtered[col] = df_filtered[col].astype(str)

            categorical = df_filtered.select_dtypes(include=[object]).columns.tolist()

            # Use Streamlit to create selection box for categorical variable
            st.header("Table 1")
            categorical_variable = st.selectbox('Select the categorical variable for grouping:', 
                                                options=categorical)
            nonnormal_variables = st.multiselect("Select any non-normally distributed variables for rank-based analysis", df_filtered.columns.tolist())

            # st.write(df_filtered.head())
            table = generate_table(df_filtered, categorical_variable, nonnormal_variables)
            # tablefmt = st.radio("Select a format for your table:", ["github", "grid", "fancy_grid", "pipe", "orgtbl", "jira", "presto", "psql", "rst", "mediawiki", "moinmoin", "youtrack", "html", "latex", "latex_raw", "latex_booktabs", "textile"])
            # st.header("Table 1")
            st.write(table.tabulate(tablefmt = "github"))
            st.write("-------")
            st.info("""Courtesy of TableOne: Tom J Pollard, Alistair E W Johnson, Jesse D Raffa, Roger G Mark;
tableone: An open source Python package for producing summary statistics
for research papers, JAMIA Open, Volume 1, Issue 1, 1 July 2018, Pages 26–31,
https://doi.org/10.1093/jamiaopen/ooy012""")
            st.write("-------")
        # Download button for Excel file
            if st.checkbox("Click to Download Your Table 1"):
                table_format = st.selectbox("Select a file format:", ["csv", "excel", "html", "latex"])

                # Save DataFrame as Excel file
                if table_format == "excel":
                    output_path = "./output/tableone_results.xlsx"
                    table.to_excel(output_path)
                    # Provide the download link
                    st.markdown(get_download_link(output_path, "xlsx"), unsafe_allow_html=True)
                    
                if table_format == "csv":
                    output_path = "./output/tableone_results.csv"
                    table.to_csv(output_path)
                    # Provide the download link
                    st.markdown(get_download_link(output_path, "csv"), unsafe_allow_html=True)
                    
                if table_format == "html":
                    output_path = "./output/tableone_results.html"
                    table.to_html(output_path)
                    # Provide the download link
                    st.markdown(get_download_link(output_path, "html"), unsafe_allow_html=True)
                    
                if table_format == "latex":
                    output_path = "./output/tableone_results.tex"
                    table.to_latex(output_path)
                    st.markdown(get_download_link(output_path, "tex"), unsafe_allow_html=True)



                # Save DataFrame as Excel file


            
    if perform_pca:
            # Create PCA plot

        pca_fig2 = perform_pca_plot(st.session_state.df)
        save_image(pca_fig2, 'pca_plot.png')
        scree_plot = create_scree_plot(st.session_state.df)
        save_image(scree_plot, 'scree_plot.png')
        
        with st.expander("What is PCA?"):
            st.write("""Principal Component Analysis, or PCA, is a method used to highlight important information in datasets that have many variables and to bring out strong patterns in a dataset. It's a way of identifying underlying structure in data.

Here's an analogy that might make it more understandable: Imagine a swarm of bees flying around in a three-dimensional space: up/down, left/right, and forward/backward. These are our original variables. Now, imagine you want to take a picture of this swarm that captures as much information as possible, but your camera can only take pictures in two dimensions. You can rotate your camera in any direction, but once you take a picture, you'll lose the third dimension. 

PCA helps us choose the best angle to take this picture. The first principal component (PC1) represents the best angle that captures the most variation in the swarm. The second principal component (PC2) is the best angle perpendicular to the first that captures the remaining variation, and so on. The idea is to minimize the information (variance) lost when we reduce dimensionality (like going from a 3D swarm to a 2D picture).

In a medical context, you might have data from thousands of genes or hundreds of physical and behavioral characteristics. Not all of these variables are independent, and many of them tend to change together. PCA allows us to represent the data in fewer dimensions that capture the most important variability in the dataset. 

Each Principal Component represents a combination of original features (like genes or patient characteristics) and can often be interpreted in terms of those features. For example, a PC might represent a combination of patient's age, blood pressure, and cholesterol level. The coefficients of the features in the PC (the "loadings") tell us how much each feature contributes to that PC.

Finally, PCA can be particularly useful in visualizing high-dimensional data. By focusing on the first two or three principal components, we can create a scatterplot of our data, potentially highlighting clusters or outliers. However, remember that this visualization doesn't capture all the variability in the data—only the variability best captured by the first few principal components.""")

with tab2:
    st.info("""N.B. This merely shows a glimpse of what is possible. Any model shown is not yet optimized and requires ML and domain level expertise.
            Yet, this is a good start to get a sense of what is possible."""
            )
    try:
        x = st.session_state.df
    except NameError:
        st.warning("First upload a CSV file or choose a demo dataset from the **Data Exploration** tab")
    else:

        # Filter categorical columns and numerical bivariate columns
        categorical_cols = st.session_state.df.select_dtypes(include=[object]).columns.tolist()

        # Add bivariate numerical columns
        numerical_bivariate_cols = [col for col in st.session_state.df.select_dtypes(include=['int64', 'float64']).columns 
                                    if st.session_state.df[col].nunique() == 2]

        # Combine the two lists and sort them
        categorical_cols = categorical_cols + numerical_bivariate_cols
        categorical_cols.sort()  # sort the list of columns

        with st.expander("Click to see your current dataset"):
            st.info("The first 5 rows:")
            st.write(st.session_state.df.head())

        st.subheader("""
        Choose the Target Column
        """)
        target_col = st.selectbox('Select a categorical column as the target:', categorical_cols)

        st.subheader("""
        Set the Target Class Value to Predict
        """)
        try:
            categories_to_predict = st.multiselect('Select one or more categories but not all. You need 2 options to predict a group, i.e, your target versus the rest.:', st.session_state.df[target_col].unique().tolist(), key = "target_categories-ml")

            # Preprocess the data and exclude the target column from preprocessing
            df_processed, included_cols, excluded_cols = preprocess(st.session_state.df.drop(columns=[target_col]), target_col)
            df_processed[target_col] = st.session_state.df[target_col]  # Include the target column back into the dataframe

            
            st.subheader("""
            Select Features to Include in the Model
            """)
            st.info(f"Available Features for your Model: {included_cols}")
            st.warning(f"Your Selected Target for Prediction: {target_col} = {categories_to_predict}")
            all_features = st.checkbox("Select all features", value=False, key="select_all_features-10")
            if all_features:
                final_columns = included_cols
            else:        
                final_columns = st.multiselect('Select features to include in your model:', included_cols, key = "columns_to_include-10")
            if len(excluded_cols) > 0:
                st.write(f"Unavailable columns for modeling: {excluded_cols}")

            # Create binary target variable based on the selected categories
            df_processed[target_col] = df_processed[target_col].apply(lambda x: 1 if x in categories_to_predict else 0)
            X = df_processed[final_columns]
            # st.write(X.head())

            # Split the dataframe into data and labels
            # List of available scaling options
            scaling_options = {
                "No Scaling": None,
                "Standard Scaling": StandardScaler(),
                "Min-Max Scaling": MinMaxScaler(),
            }

            # List of available normalization options
            normalization_options = {
                "No Normalization": None,
                "L1 Normalization": "l1",
                "L2 Normalization": "l2",
            }
            scaling_or_norm = st.checkbox("Scaling or Normalization?", value=False, key="scaling_or_norm-10")
            # User selection for scaling option
            if scaling_or_norm == True:
                scaling_option = st.selectbox("Select Scaling Option", list(scaling_options.keys()))
                # User selection for normalization option
                normalization_option = st.selectbox("Select Normalization Option", list(normalization_options.keys()))

            # Apply selected scaling and normalization options to the features
            

                if scaling_option != "No Scaling":
                    scaler = scaling_options[scaling_option]
                    X = scaler.fit_transform(X)

                if normalization_option != "No Normalization":
                    normalization_type = normalization_options[normalization_option]
                    X = normalize(X, norm=normalization_type)
            # X = df_processed.drop(columns=[target_col])
            # X = df_processed[final_columns]
            y = df_processed[target_col]

            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # pca_check = st.checkbox("PCA?", value=False, key="pca_check-10")
            # if pca_check == True:
            #     n_neighbors = 3
            #     random_state = 0
            #     dim = len(X[0])
            #     n_classes = len(np.unique(y))

            #     # Reduce dimension to 2 with PCA
            #     pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=random_state))

            #     # Reduce dimension to 2 with LinearDiscriminantAnalysis
            #     lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))

            #     # Reduce dimension to 2 with NeighborhoodComponentAnalysis
            #     nca = make_pipeline(
            #         StandardScaler(),
            #         NeighborhoodComponentsAnalysis(n_components=2, random_state=random_state),
            #     )

            #     # Use a nearest neighbor classifier to evaluate the methods
            #     knn = KNeighborsClassifier(n_neighbors=n_neighbors)

            #     # Make a list of the methods to be compared
            #     dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]

            #     # plt.figure()
            #     for i, (name, model) in enumerate(dim_reduction_methods):
            #         plt.figure()
            #         # plt.subplot(1, 3, i + 1, aspect=1)

            #         # Fit the method's model
            #         model.fit(X_train, y_train)

            #         # Fit a nearest neighbor classifier on the embedded training set
            #         knn.fit(model.transform(X_train), y_train)

            #         # Compute the nearest neighbor accuracy on the embedded test set
            #         acc_knn = knn.score(model.transform(X_test), y_test)

            #         # Embed the data set in 2 dimensions using the fitted model
            #         X_embedded = model.transform(X)

            #         # Plot the projected points and show the evaluation score
            #         plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap="Set1")
            #         plt.title(
            #             "{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn)
            #         )
            #     fig = plt.show()
            #     st.pyplot(fig)
        except:
            st.warning("Please select a target column first or pick a dataset with a target column avaialble.")        

        st.subheader("""
        Choose the Machine Learning Model
        """)
        model_option = st.selectbox(
            "Which machine learning model would you like to use?",
            ("Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting Machines (GBMs)", "Support Vector Machines (SVMs)", "Neural Network")
        )
        perform_shapley = st.checkbox("Include a Shapley Force Plot", value=False, key="perform_shapley-10")
        if perform_shapley == True:
            st.warning("Shapley interpretation of the model is computationally expensive for some models and may take a while to run. Please be patient")
        if st.button("Predict"):
            if model_option == "Logistic Regression":
                model = LogisticRegression()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                y_scores = model.predict_proba(X_test)[:, 1]
                    
                with st.expander("What is logistic regression?"):
                    st.write("""
Logistic regression is a statistical model commonly used in the field of medicine to predict binary outcomes - such as whether a patient has a disease (yes/no), whether a patient survived or not after a treatment (survived/did not survive), etc.

Logistic regression, like linear regression, establishes a relationship between the predictor variables (such as patient's age, weight, smoking history) and the target variable (e.g., presence or absence of a disease). However, unlike linear regression which predicts a continuous outcome, logistic regression predicts the probability of an event occurring, which is perfect for binary (two-category) outcomes.

Here's a simplified step-by-step breakdown:

1. **Collect and Prepare Your Data**: This involves gathering medical data that includes both the outcome (what you want to predict) and predictor variables (information you will use to make the prediction).

2. **Build the Model**: Logistic regression uses a mathematical formula that looks somewhat similar to the formula for a line in algebra (y = mx + b), but it's modified to predict probabilities. The formula takes your predictors and calculates the "log odds" of the event occurring.

3. **Interpret the Model**: The coefficients (the values that multiply the predictors) in the logistic regression model represent the change in the log odds of the outcome for a one-unit increase in the predictor variable. For example, if age is a predictor and its coefficient is 0.05, it means that for each one year increase in age, the log odds of the disease occurring (assuming all other factors remain constant) increase by 0.05. Because these are "log odds", the relationship between the predictors and the probability of the outcome isn't a straight line, but a curve that can't go below 0 or above 1.

4. **Make Predictions**: You can input a new patient's information into the logistic regression equation, and it will output the predicted probability of the outcome. For example, it might predict a patient has a 75% chance of having a disease. You can then convert this into a binary outcome by setting a threshold, such as saying any probability above 50% will be considered a "yes."

Remember that logistic regression, while powerful, makes several assumptions. It assumes a linear relationship between the log odds of the outcome and the predictor variables, it assumes that errors are not measured and that there's no multicollinearity (a high correlation among predictor variables). As with any model, it's also only as good as the data you feed into it.

In the medical field, logistic regression can be a helpful tool to predict outcomes and identify risk factors. However, it's important to understand its assumptions and limitations and to use clinical judgment alongside the model's predictions.""")

                display_metrics(y_test, predictions, y_scores)
                # After training the logistic regression model, assuming the model's name is "model"

                coeff = model.coef_[0]
                features = X_train.columns

                equation = "Logit(P) = " + str(model.intercept_[0])
                
                for c, feature in zip(coeff, features):
                    equation += " + " + str(c) + " * " + feature

                st.write("The equation of the logistic regression model is:")
                st.write(equation)
                
                if perform_shapley == True:                     # shapley explanation
                
                    # Scale the features
                    with st.expander("What is a Shapley Force Plot?"):
                        st.markdown(shapley_explanation)  
                    with st.spinner("Performing Analysis for the Shapley Force Plot..."):
                        # Standardize the features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        # shapley explanation using KernelExplainer
                        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train_scaled, 100))
                        shap_values = explainer.shap_values(X_test_scaled)

                        # Sort features by absolute contribution for the first instance in the test set
                        sorted_indices = np.argsort(np.abs(shap_values[1][0]))[::-1]
                        sorted_shap_values = shap_values[1][0][sorted_indices]
                        sorted_feature_names = X_test.columns[sorted_indices]

                        # Create a DataFrame to display sorted features and their shapley values
                        sorted_features_df = pd.DataFrame({
                            'Feature': sorted_feature_names,
                            'Shapley_Value': sorted_shap_values
                        })

                        # Display the sorted features DataFrame in Streamlit
                        st.table(sorted_features_df)

                        # Generate and display the sorted force plot
                        shap_html = shap.force_plot(explainer.expected_value[1], sorted_shap_values, sorted_feature_names, show=False)
                        shap.save_html("sorted_shap_plot.html", shap_html)
                        with open("sorted_shap_plot.html", "r") as f:
                            st.components.v1.html(f.read(), height=500)


            elif model_option == "Decision Tree":
                model = DecisionTreeClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                y_scores = model.predict_proba(X_test)[:, 1]
                with st.expander("What is a decision tree?"):
                    st.write("""
A decision tree is a type of predictive model that you can think of as similar to the flowcharts sometimes used in medical diagnosis. They're made up of nodes (decision points) and branches (choices or outcomes), and they aim to predict an outcome based on input data.

Here's how they work:

1. **Start at the root node**: This is the first decision that needs to be made and it's based on one of your input variables. For instance, in a medical context, this might be a question like, "Is the patient's temperature above 100.4 degrees Fahrenheit?"

2. **Follow the branch for your answer**: If the answer is "yes," follow the branch for "yes," and if it's "no," follow the branch for "no."

3. **Make the next decision**: Each branch leads to another node, where another decision will be made based on another variable. Maybe this time it's, "Does the patient have a cough?"

4. **Continue until you reach a leaf node**: Leaf nodes are nodes without any further branches. They represent the final decisions and are predictions of the outcome. In a binary outcome scenario, leaf nodes could represent "disease" or "no disease."

The decision tree "learns" from data by splitting the data at each node based on what would provide the most significant increase in information (i.e., the best separation of positive and negative cases). For instance, if patients with a certain disease often have a fever, the model might learn to split patients based on whether they have a fever.

While decision trees can be powerful and intuitive tools, there are a few caveats to keep in mind:

- **Overfitting**: If a tree is allowed to grow too deep (too many decision points), it may start to fit not just the underlying trends in the data, but also the random noise. This means it will perform well on the data it was trained on, but poorly on new data.

- **Instability**: Small changes in the data can result in a very different tree. This can be mitigated by using ensemble methods, which combine many trees together (like a random forest).

- **Simplicity**: Decision trees make very simple, linear cuts in the data. They can struggle with relationships in the data that are more complex.

Overall, decision trees can be an excellent tool for understanding and predicting binary outcomes from medical data. They can handle a mixture of data types, deal with missing data, and the results are interpretable and explainable. Just like with any medical test, though, the results should be interpreted with care and in the context of other information available."""
                    )
                display_metrics(y_test, predictions, y_scores)
                if perform_shapley == True:                     # shapley explanation
                
                    # Scale the features
                    with st.expander("What is a Shapley Force Plot?"):
                        st.markdown(shapley_explanation)  
                    with st.spinner("Performing Analysis for the Shapley Force Plot..."):

                        # shapley explanation using TreeExplainer
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test)

                        # Sort features by absolute contribution for the first instance in the test set
                        sorted_indices = np.argsort(np.abs(shap_values[1][0]))[::-1]
                        sorted_shap_values = shap_values[1][0][sorted_indices]
                        sorted_feature_names = X_test.columns[sorted_indices]

                        # Create a DataFrame to display sorted features and their shapley values
                        sorted_features_df = pd.DataFrame({
                            'Feature': sorted_feature_names,
                            'Shapley_Value': sorted_shap_values
                        })

                        # Display the sorted features DataFrame in Streamlit
                        st.table(sorted_features_df)

                        # Generate and display the sorted force plot
                        shap_html = shap.force_plot(explainer.expected_value[1], sorted_shap_values, sorted_feature_names, show=False)
                        shap.save_html("sorted_shap_plot.html", shap_html)
                        with open("sorted_shap_plot.html", "r") as f:
                            st.components.v1.html(f.read(), height=500)

            elif model_option == "Random Forest":
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                y_scores = model.predict_proba(X_test)[:, 1]
                with st.expander("What is a random forest?"):
                    st.write("""
Random Forest is a type of machine learning model that is excellent for making predictions (both binary and multi-class) based on multiple input variables, which can be both categorical (like gender: male or female) and numerical (like age or blood pressure).

Imagine you have a patient and you have collected a lot of data about them - age, weight, cholesterol level, blood pressure, whether or not they smoke, etc. You want to predict a binary outcome: will they have a heart attack in the next 10 years or not? 

A Random Forest works a bit like a team of doctors, each of whom asks a series of questions to make their own diagnosis (or prediction). These doctors are analogous to "decision trees" - the building blocks of a Random Forest.

Here's a simplified breakdown of how it works:

1. **Building Decision Trees**: Each "doctor" (or decision tree) in the Random Forest gets a random subset of patients' data. They ask questions like, "Is the patient's age over 60?", "Is their cholesterol level over 200?". Depending on the answers, they follow different paths down the tree, leading to a final prediction. The tree is constructed in a way that the most important questions (those that best split the patients according to the outcome) are asked first.

2. **Making Predictions**: To make a prediction for a new patient, each decision tree in the Random Forest independently makes a prediction. Essentially, each tree "votes" for the outcome it thinks is most likely (heart attack or no heart attack).

3. **Combining the Votes**: The Random Forest combines the votes from all decision trees. The outcome that gets the most votes is the Random Forest's final prediction. This is like asking a team of doctors for their opinions and going with the majority vote.

One of the main strengths of Random Forest is that it can handle complex data with many variables and it doesn't require a lot of data preprocessing (like scaling or normalizing data). Also, it is less prone to "overfitting" compared to individual decision trees. Overfitting is when a model learns the training data too well, to the point where it captures noise and performs poorly when predicting outcomes for new, unseen data.

However, it's important to note that while Random Forest often performs well, it can be somewhat of a "black box", meaning it can be hard to understand why it's making the predictions it's making. It's always crucial to validate the model's predictions against your medical knowledge and context."""
                    )
                display_metrics(y_test, predictions, y_scores)
                if perform_shapley == True:                     # shapley explanation
                
                    # Scale the features
                    with st.expander("What is a Shapley Force Plot?"):
                        st.markdown(shapley_explanation)  
                    with st.spinner("Performing Analysis for the Shapley Force Plot..."):
                        # shapley explanation using TreeExplainer
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test)

                        # Sort features by absolute contribution for the first instance in the test set
                        sorted_indices = np.argsort(np.abs(shap_values[1][0]))[::-1]
                        sorted_shap_values = shap_values[1][0][sorted_indices]
                        sorted_feature_names = X_test.columns[sorted_indices]

                        # Create a DataFrame to display sorted features and their shapley values
                        sorted_features_df = pd.DataFrame({
                            'Feature': sorted_feature_names,
                            'Shapley_Value': sorted_shap_values
                        })

                        # Display the sorted features DataFrame in Streamlit
                        st.table(sorted_features_df)

                        # Generate and display the sorted force plot
                        shap_html = shap.force_plot(explainer.expected_value[1], sorted_shap_values, sorted_feature_names, show=False)
                        shap.save_html("sorted_shap_plot.html", shap_html)
                        with open("sorted_shap_plot.html", "r") as f:
                            st.components.v1.html(f.read(), height=500)
                                
                
            elif model_option == "Gradient Boosting Machines (GBMs)":
                model = GradientBoostingClassifier()
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                y_scores = model.predict_proba(X_test)[:, 1]
                with st.expander("What is a gradient boosting machine?"):
                    st.write("""Gradient Boosting Machines, like Random Forests, are a type of machine learning model that is good at making predictions based on multiple input variables. These variables can be both categorical (like patient sex: male or female) and numerical (like age, heart rate, etc.). 

Again, suppose we're trying to predict a binary outcome: will this patient develop diabetes in the next five years or not?

A GBM also uses decision trees as its building blocks, but there's a crucial difference in how GBM combines these trees compared to Random Forests. Rather than having each tree independently make a prediction and then voting on the final outcome, GBMs build trees in sequence where each new tree is trying to correct the mistakes of the combined existing trees.

Here's a simplified breakdown of how it works:

1. **Building the First Tree**: A single decision tree is built to predict the outcome based on the input variables. However, this tree is usually very simple and doesn't do a great job at making accurate predictions.

2. **Building Subsequent Trees**: New trees are added to the model. Each new tree is constructed to correct the errors made by the existing set of trees. It does this by predicting the 'residual errors' of the previous ensemble of trees. In other words, it tries to predict how much the current model is 'off' for each patient.

3. **Combining the Trees**: The predictions from all trees are added together to make the final prediction. Each tree's contribution is 'weighted', so trees that do a better job at correcting errors have a bigger say in the final prediction.

GBMs are a very powerful method and often perform exceptionally well. Like Random Forests, they can handle complex data with many variables. But they also have a few additional strengths:

- GBMs can capture more complex patterns than Random Forests because they build trees sequentially, each learning from the last.

- GBMs can also give an estimate of the importance of each variable in making predictions, which can be very useful in understanding what's driving your predictions.

However, GBMs do have their challenges:

- They can be prone to overfitting if not properly tuned. Overfitting happens when your model is too complex and starts to capture noise in your data rather than the true underlying patterns.

- They can also be more computationally intensive than other methods, meaning they might take longer to train, especially with larger datasets.

Just like with any model, it's crucial to validate the model's predictions with your medical knowledge and consider the context. It's also important to remember that while GBMs can make very accurate predictions, they don't prove causation. They can identify relationships and patterns in your data, but they can't tell you why those patterns exist.""")
                    
                display_metrics(y_test, predictions, y_scores)
                if perform_shapley == True:                     # shapley explanation
                
                    # Scale the features
                    with st.expander("What is a Shapley Force Plot?"):
                        st.markdown(shapley_explanation)  
                    with st.spinner("Performing Analysis for the Shapley Force Plot..."):


                        # shapley explanation
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_test)

                        # Check if shap_values is a list (multi-class) or a single array (binary classification or regression)
                        if isinstance(shap_values, list):
                            shap_values_for_class = shap_values[1]  # Assuming you're interested in the second class
                        else:
                            shap_values_for_class = shap_values

                        # Sort features by absolute contribution for the first instance in the test set
                        sorted_indices = np.argsort(np.abs(shap_values_for_class[0]))[::-1]
                        sorted_shap_values = shap_values_for_class[0][sorted_indices]
                        sorted_feature_names = X_test.columns[sorted_indices]

                        # Create a DataFrame to display sorted features and their shapley values
                        sorted_features_df = pd.DataFrame({
                            'Feature': sorted_feature_names,
                            'Shapley_Value': sorted_shap_values
                        })

                        # Display the sorted features DataFrame in Streamlit
                        st.table(sorted_features_df)

                        # Generate and display the sorted force plot
                        shap_html = shap.force_plot(explainer.expected_value, sorted_shap_values, sorted_feature_names, show=False)
                        shap.save_html("sorted_shap_plot.html", shap_html)
                        with open("sorted_shap_plot.html", "r") as f:
                            st.components.v1.html(f.read(), height=500)




            elif model_option == "Support Vector Machines (SVMs)":
                model = svm.SVC(probability=True)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                y_scores = model.predict_proba(X_test)[:, 1]
                with st.expander("What is a support vector machine?"):
                    st.write("""
Support Vector Machines are a type of machine learning model that can be used for both regression and classification tasks. They can handle both numerical and categorical input variables. In the context of predicting a binary outcome in medical data - let's stick with the example of predicting whether a patient will develop diabetes or not in the next five years - an SVM is a classification tool.

Here's a simplified explanation:

1. **Building the Model**: The SVM algorithm tries to find a hyperplane, or a boundary, that best separates the different classes (in our case, 'will develop diabetes' and 'will not develop diabetes'). This boundary is chosen to be the one that maximizes the distance between the closest points (the "support vectors") in each class, which is why it's called a "Support Vector Machine".

2. **Making Predictions**: Once this boundary is established, new patients can be classified by where they fall in relation to this boundary. If a new patient's data places them on the 'will develop diabetes' side of the boundary, the SVM predicts they will develop diabetes.

Here are some strengths and challenges of SVMs:

Strengths:
- SVMs can model non-linear decision boundaries, and there are many kernels to choose from. This can make them more flexible in capturing complex patterns in the data compared to some other methods.
- They are also fairly robust against overfitting, especially in high-dimensional space.

Challenges:
- However, SVMs are not very easy to interpret compared to models like decision trees or logistic regression. The boundaries they produce can be complex and not easily explainable in terms of the input variables.
- SVMs can be inefficient to train with very large datasets, and they require careful preprocessing of the data and tuning of the parameters.

As with any machine learning model, while an SVM can make predictions about patient health, it's crucial to validate these predictions with medical expertise. Furthermore, an SVM can identify relationships in data, but it doesn't explain why these relationships exist. As always, correlation doesn't imply causation.""")
                display_metrics(y_test, predictions, y_scores)
                if perform_shapley == True:
                    with st.expander("What is a Shapley Force Plot?"):
                        st.markdown(shapley_explanation)  
                    with st.spinner("Performing Analysis for the Shapley Force Plot..."):
                    
                        # shapley explanation using KernelExplainer for SVM
                        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
                        shap_values = explainer.shap_values(X_test)

                        # Check if shap_values is a list (multi-class) or a single array (binary classification or regression)
                        if isinstance(shap_values, list):
                            shap_values_for_class = shap_values[1]  # Assuming you're interested in the second class
                        else:
                            shap_values_for_class = shap_values

                        # Sort features by absolute contribution for the first instance in the test set
                        sorted_indices = np.argsort(np.abs(shap_values_for_class[0]))[::-1]
                        sorted_shap_values = shap_values_for_class[0][sorted_indices]
                        sorted_feature_names = X_test.columns[sorted_indices]

                        # Create a DataFrame to display sorted features and their shapley values
                        sorted_features_df = pd.DataFrame({
                            'Feature': sorted_feature_names,
                            'Shapley_Value': sorted_shap_values
                        })

                        # Display the sorted features DataFrame in Streamlit
                        st.table(sorted_features_df)

                        # Generate and display the sorted force plot
                        shap_html = shap.force_plot(explainer.expected_value[1], sorted_shap_values, sorted_feature_names, show=False)
                        shap.save_html("sorted_shap_plot.html", shap_html)
                        with open("sorted_shap_plot.html", "r") as f:
                            st.components.v1.html(f.read(), height=500)
                
            elif model_option == "Neural Network":
                model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu')
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                y_scores = model.predict_proba(X_test)[:, 1]
                with st.expander("What is a neural network?"):
                    st.write("""
A neural network is a type of machine learning model inspired by the structure and function of the human brain's neural network. It is excellent for solving complex problems and making predictions based on historical data.

Just like the human brain consists of interconnected neurons, a neural network consists of interconnected artificial neurons called "nodes" or "neurons". These neurons are organized in layers - an input layer, one or more hidden layers, and an output layer. Each neuron takes input from the previous layer, performs a mathematical operation on the input, and passes the result to the next layer.

Here's a simplified breakdown of how a neural network works:

1. **Feedforward**: The input layer receives the input data, which can be numerical or categorical variables. Each neuron in the hidden layers and the output layer performs a weighted sum of the inputs, applies an activation function, and passes the result to the next layer. This process is called feedforward.

2. **Activation Function**: The activation function introduces non-linearity to the neural network, allowing it to learn and model complex relationships in the data. Common activation functions include sigmoid, tanh, and ReLU.

3. **Backpropagation**: After the feedforward process, the neural network compares its predictions to the actual values and calculates the prediction error. It then adjusts the weights and biases of the neurons in the network through a process called backpropagation. This iterative process continues until the neural network reaches a satisfactory level of accuracy.

Neural networks can be used for a wide range of tasks, including regression, classification, and even more complex tasks like image and speech recognition. They have been successfully applied in various domains, including medicine, finance, and natural language processing.

However, it's important to note that neural networks are computationally intensive and require a large amount of training data to generalize well. Additionally, hyperparameter tuning and regularization techniques may be necessary to prevent overfitting and improve performance.
            """
        )
                display_metrics(y_test, predictions, y_scores)
                
                if perform_shapley == True:
                    with st.expander("What is a Shapley Force Plot?"):
                        st.markdown(shapley_explanation)  
                    
                    with st.spinner("Performing Shapley Analysis..."):
                    
                        # shapley explanation using KernelExplainer for MLP
                        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
                        shap_values = explainer.shap_values(X_test)

                        # Check if shap_values is a list (multi-class) or a single array (binary classification or regression)
                        if isinstance(shap_values, list):
                            shap_values_for_class = shap_values[1]  # Assuming you're interested in the second class
                        else:
                            shap_values_for_class = shap_values

                        # Sort features by absolute contribution for the first instance in the test set
                        sorted_indices = np.argsort(np.abs(shap_values_for_class[0]))[::-1]
                        sorted_shap_values = shap_values_for_class[0][sorted_indices]
                        sorted_feature_names = X_test.columns[sorted_indices]

                        # Create a DataFrame to display sorted features and their shapley values
                        sorted_features_df = pd.DataFrame({
                            'Feature': sorted_feature_names,
                            'Shapley_Value': sorted_shap_values
                        })

                        # Display the sorted features DataFrame in Streamlit
                        st.table(sorted_features_df)

                        # Generate and display the sorted force plot
                        shap_html = shap.force_plot(explainer.expected_value[1], sorted_shap_values, sorted_feature_names, show=False)
                        shap.save_html("sorted_shap_plot.html", shap_html)
                        with open("sorted_shap_plot.html", "r") as f:
                            st.components.v1.html(f.read(), height=500)

                


        