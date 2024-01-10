import streamlit as st
import plotly.express as px
import pandas as pd
# import openai
# import geopandas as gpd

# Page settings
st.set_page_config(
    page_title="ProsperaLima: Illuminating Pathways to Urban Excellence",
    page_icon="🏙️",
    layout="wide",
)

# Define a list of file paths
file_paths = [
    r"Datasets/Investment Projects of Lima.csv"
]

@st.cache_data
# Read data from a file path
def get_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


df = get_data(file_paths[0])


# Cost-to-Earnings Ratio
df['Cost_to_Earnings_Ratio'] = df['UPDATED_COST'] / df['ACCUMULATED_EARNINGS_AS_OF_2021']
# Budget Execution Rate for 2022
df['Budget_Execution_Rate_2022'] = df['AMOUNT_YEAR_2022'] / df['PIM_2022']
# 'Success' column indicating project success (1 for success, 0 for failure)
df['Success'] = (df['AMOUNT_YEAR_2025'] > 0).astype(int)
# Assuming 'PIM_2022' is the planned budget for the year 2022
# Calculate budget adherence for Formulating Units
df['Budget_Adherence_Formulating_Unit'] = df['AMOUNT_YEAR_2022'] / df['PIM_2022']
# Calculate budget adherence for Executing Units
df['Budget_Adherence_Executing_Unit'] = df['AMOUNT_YEAR_2022'] / df['PIM_2022']

def display_section(Heading,Content):
    st.markdown(
        f"""
        <div style="
            background-color: #90caf9;
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
col1, col2 = st.columns([2, 8])

# Create a session state
if "button_state" not in st.session_state:
    st.session_state.button_state = False

with col1:
    Obj = "Objective"
    Cont_1 = "To gain insights into the investment projects of the Regional Government of Lima and leverage the findings to contribute to the improvement of Lima's economy and overall quality of life."
    display_section(Obj, Cont_1)

with col2:
    st.title("Building Tomorrow: Unveiling the Future Through Regional Investments")

    # Create a button to display population data
    button1 = st.button("Projects Data")
    if button1:
        # Toggle the button state
        st.session_state.button_state = not st.session_state.button_state

    # Display the sample of the population data if the button is pressed
    if st.session_state.button_state:
        # Display the sample of the population data
        st.subheader("Investment Projects Dataset of Department of Lima:")
        st.dataframe(df)

    else:
        tab1 = st.tabs(["Investment Projects 2021"])

        # 1. Understand the distribution of investment projects across different provinces.
        province_counts = df['PROVINCE'].value_counts()
        fig = px.bar(province_counts, x=province_counts.index, y=province_counts.values,
                     labels={'value': 'Number of Investment Projects', 'PROVINCE': 'Province'})
        fig.update_layout(title='Distribution of Investment Projects Across Provinces', xaxis_title='Provinces',
                          yaxis_title='Number of Investment Projects')
        st.plotly_chart(fig)
        st.write("'HUARAL' has the highest number of projects (56), followed by 'HUAURA' (46) and 'SIN PROVINCIA' (46)")

        # 2. Unit Performance
        # Calculate success rates for Executing Units
        executing_unit_success_rate = df.groupby('INVESTMENT_EXECUTING_UNIT')['Success'].mean().sort_values(
            ascending=True)
        fig_2 = px.bar(executing_unit_success_rate, x=executing_unit_success_rate.values,
                       y=executing_unit_success_rate.index,
                       title='Success Rate by Executing Unit')
        fig_2.update_layout(xaxis_title='Success Rate', yaxis_title='Executing Unit')
        st.plotly_chart(fig_2)
        st.write("'National Intelligence Service (SIN)' and 'REGIONAL MANAGEMENT OF SOCIAL DEVELOPMENT' have high success rates, while other units vary in their success rates.")

        # 3. Function-specific Analysis
        # calculate the count of investments in each function
        function_distribution = df.groupby('FUNCTION')['INVESTMENT_NAME'].count().sort_values(ascending=True)
        fig_3 = px.bar(function_distribution, x=function_distribution.values, y=function_distribution.index,
                       title='Function-specific Investment Distribution')
        fig_3.update_layout(xaxis_title='Number of Investments', yaxis_title='Functions')
        st.plotly_chart(fig_3)
        st.write("'EDUCATION' and 'AGRICULTURE AND LIVESTOCK' are the most common functions, suggesting a focus on these areas. Other functions like 'PUBLIC ORDER AND SECURITY'and 'HEALTH' also receive significant investments.")

        # Conclusion
        st.title("Conclusion")
        # Conclusion content
        conclusion_text = """
        **Investment Distribution:**
        - Investments are concentrated in provinces such as Huaral, Huaura, and Cañete.
        - Formulating and executing units vary in their success rates, with some achieving a 100% success rate.

        **Sectoral Focus:**
        - Education and agriculture are the predominant sectors receiving investments, emphasizing the government's focus on these areas.

        **Recommendations:**
        - Strategic alignment, diversification of investments, knowledge sharing, capacity building, and stakeholder engagement are recommended strategies.
        - The Regional Government of Lima should adopt a culture of continuous improvement, leveraging data-driven insights for informed decision-making.
        """
        # Display the conclusion
        st.markdown(conclusion_text)



