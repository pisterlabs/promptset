import streamlit as st
import plotly.express as px
import pandas as pd
# import openai
import geopandas as gpd

# Page settings
st.set_page_config(
    page_title="ProsperaLima: Illuminating Pathways to Urban Excellence",
    page_icon="🏙️",
    layout="wide",
)

# Define a list of file paths
file_paths = [
    r"Datasets/Monthly average of Private sector Workers-2021.csv"
]

@st.cache_data
# Read data from a file path
def get_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


df = get_data(file_paths[0])

def display_section(Heading,Content):
    st.markdown(
        f"""
        <div style="
            background-color: #70d1d0;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        ">
        <h3>{Heading}</h3>
        <p>{Content}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# creating column for homepage
col1, col2 = st.columns([2, 6])

# Create a session state
if "button_state" not in st.session_state:
    st.session_state.button_state = False

with col1:
    Obj = "Objective"
    Cont_1 = "To gain insights into the monthly average of private sector workers in Lima, categorized by different pension regimes and distributed across districts." \
             "By understanding the distribution of workers among various pension schemes, we aim to identify patterns, trends, and potential areas for improvement in Lima's workforce management and economic development."
    display_section(Obj, Cont_1)

    Pension = "Private Pension System (SPP)"
    Cont_2 = "It is regulated and supervised by the Superintendency of Banking, Insurance and AFP, which aims to preserve the interests of SPP members.\n" \
             "SPP INTEGRA: Integra Private Pension Fund, SPP HORIZONTE: Horizonte Private Pension Fund, SPP PROFUTURO: Profuturo Private Pension Fund, SPP PRIMA: Prima Private Pension Fund, SPP HABITAT: Habitat Private Pension Fund.\n " \
             "These are the names of the five private pension fund administrators (AFPs) operating in Peru. Each AFP manages its own investment portfolio and offers different investment options to its members."
    display_section(Pension, Cont_2)

with col2:
    st.title("Employment Trends and Pension Scheme Distribution: Lima 2021")

    # Create a button to display population data
    button1 = st.button("Workers Data")
    if button1:
        # Toggle the button state
        st.session_state.button_state = not st.session_state.button_state

    # Display the sample of the population data if the button is pressed
    if st.session_state.button_state:
        # Display the sample of the population data
        st.subheader("Private Sector Workers Dataset of Lima Province:")
        st.dataframe(df[['UBIGEO_CODE', 'DISTRICTS',
       'DECREE_LAW_19990_NATIONAL_ONP_PENSION_SYSTEM', 'DECRETO_LEY_20530 ',
       'FISHERMEN_SOCIAL_SECURITY_BENEFITS_FUND', 'MILITARY_PENSION_FUND',
       'POLICE_PENSION_FUND',
       'LAW_29903_NATIONAL_SYSTEM_OF_INDEPENDENT_PENSIONS',
       'LAW_30003_SPECIAL_PENSIONS_SCHEME_FOR_FISHERMEN', 'SPP_INTEGRA',
       'SPP_HORIZONTE', 'SPP_PROFUTURO', 'SPP_PRIMA', 'SPP_HABITAT',
       'PENDING_CHOICE_OF_PENSION_PLAN', 'NO_PENSION_PLAN_DOES_NOT_APPLY',
       'NOT_DETERMINED']])

    else:
        tab1 = st.tabs(["Private sector Workers"])
        # 1. The most and least subscribed pension schemes
        # Identify the most and least subscribed pension schemes
        pension_columns = ['SPP_INTEGRA', 'SPP_HORIZONTE', 'SPP_PROFUTURO', 'SPP_PRIMA', 'SPP_HABITAT']
        pension_distribution = df[pension_columns]

        # Sum the number of workers for each pension scheme
        pension_distribution_totals = pension_distribution.sum().sort_values()

        # Create a Plotly bar chart
        fig_1 = px.bar(
            x=pension_distribution_totals.values,
            y=pension_distribution_totals.index,
            orientation='h',
            labels={'x': 'Number of Workers', 'y': 'Pension Schemes'},
            title='Distribution of Private Sector Workers Across Pension Schemes',
            color_discrete_sequence=['teal'] * len(pension_distribution_totals),
        )
        # Display the chart
        st.plotly_chart(fig_1)


        # 2. District-wise Workforce Assessment
        pension_scheme_columns = df.columns[2:]
        # Sum the values across pension scheme columns to get the total number of workers
        df['Total_Workers'] = df[pension_scheme_columns].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        # Group the data by districts and calculate the total number of workers in each district
        district_workforce = df.groupby('DISTRICTS')['Total_Workers'].sum().reset_index()
        # Sort the districts based on the total number of workers
        district_workforce_sorted = district_workforce.sort_values(by='Total_Workers', ascending=True)
        # Create a Plotly bar chart
        fig_2 = px.bar(
            district_workforce_sorted,
            x='Total_Workers',
            y='DISTRICTS',
            orientation='h',
            labels={'x': 'Number of Workers', 'y': 'Districts'},
            title='Distribution of Private Sector Workers Across Lima\'s Districts',
            color='Total_Workers',
            color_continuous_scale='teal',
        )
        # Display the chart using Streamlit
        st.plotly_chart(fig_2)

        # 3. Exploring if certain demographic groups prefer specific pension schemes

        # Calculate the total number of workers in each district
        district_worker_totals = df.groupby('DISTRICTS').sum().sum(axis=1).sort_values(ascending=False)
        # Select the top 5 districts
        top_5_districts = district_worker_totals.head(5).index
        # Filter the dataframe for the top 5 districts
        df_top_5 = df[df['DISTRICTS'].isin(top_5_districts)]
        # Melt the dataframe for visualization
        melted_df_top_5 = pd.melt(df_top_5, id_vars=['DISTRICTS'],
                                  value_vars=['SPP_INTEGRA', 'SPP_HORIZONTE', 'SPP_PROFUTURO', 'SPP_PRIMA',
                                              'SPP_HABITAT'],
                                  var_name='Preferred_Pension_Scheme', value_name='Number_of_Workers')
        # Create a Plotly bar chart
        fig_top_5 = px.bar(
            melted_df_top_5,
            x='DISTRICTS',
            y='Number_of_Workers',
            color='Preferred_Pension_Scheme',
            title='Pension Scheme Preference in Top 5 Districts',
            labels={'DISTRICTS': 'Districts', 'Number_of_Workers': 'Number of Workers'},
        )
        # Display the chart
        st.plotly_chart(fig_top_5)

        # 4. Pension Scheme Participation and Employment Rates
        st.subheader('"Pension Scheme Participation and Employment Rates in Top Districts:"')
        numeric_columns = df[pension_scheme_columns].select_dtypes(include=['int', 'float']).columns
        df['Overall_Pension_Participation'] = df[numeric_columns].sum(axis=1)
        df['Employment_Rate'] = (df['Total_Workers'] / df['Total_Workers'].sum()) * 100
        threshold = 100000
        districts = df[abs(df['Overall_Pension_Participation'] - df['Employment_Rate']) > threshold]
        result = (districts[['DISTRICTS', 'Overall_Pension_Participation', 'Employment_Rate']]
                  .sort_values(by='Overall_Pension_Participation',ascending=False)
                  .astype({'Overall_Pension_Participation': str, 'Employment_Rate': str}))
        st.table(result)





