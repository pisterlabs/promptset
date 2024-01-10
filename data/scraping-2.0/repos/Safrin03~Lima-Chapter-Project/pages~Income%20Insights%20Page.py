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
    r"Datasets/Population.csv",
    r"Datasets/Pucusana District Income Data.csv"
]


@st.cache_data
def get_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path) 

df = get_data(file_paths[0])
df2 = get_data(file_paths[1])

# List to store DataFrames from individual chunks of San Juan de Miraflores
chunks = []
num_chunks = 2 
for i in range(1, num_chunks + 1):
    chunk_path = f'Datasets/San Juan De Miraflores_Income_{i}.csv'
    chunk_df = pd.read_csv(chunk_path)
    chunks.append(chunk_df)
df1 = pd.concat(chunks, ignore_index=True)

# List to store DataFrames from individual chunks of Santa Marita
chunks_ = []
num_chunks_ = 6  
for i in range(1, num_chunks_ + 1):
    chunk_path = f'Datasets/Santa Anita_Income_{i}.csv'
    chunk_df = pd.read_csv(chunk_path)
    chunks_.append(chunk_df)
df3 = pd.concat(chunks_, ignore_index=True)



# Col1 style 
def display_section(Heading,Content):
    st.markdown(
        f"""
        <div style="
            background-color: #1da1f2;
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


# creating column
col1, col2 = st.columns([2, 8])

# Create a session state
if "button_state_1" not in st.session_state:
    st.session_state.button_state_1 = False
if "button_state_2" not in st.session_state:
    st.session_state.button_state_2 = False
if "button_state_3" not in st.session_state:
    st.session_state.button_state_3 = False

with col1:
    Obj = "Objective"
    Cont_1 = 'To gain insights into the revenue collected by the "District Municipality of Santa Juan de Miraflores in Lima Province for the year 2022", ' \
             '"District Municipality of Santa Anita for the year 2023 (January to May)" and "District Municipality of Pucusana for the year 2022 (Jan to Sept)". By exploring the patterns and details within the dataset, the aim is to better understand the financial dynamics, identify trends, and derive actionable insights to contribute to the improvement of Lima\'s economy and quality of life.'
    display_section(Obj, Cont_1)

with col2:
    st.title("District Income Insights: Unveiling Lima's Financial Landscape")

    # Buttons for Dataset
    button_col1, button_col2, button_col3 = st.columns(3)

    if button_col1.clicked or button_col2.clicked or button_col3.clicked:
        st.session_state["button_state"] = not st.session_state.get("button_state", False)

    # Display the sample of the data if the button is pressed
    if button_col1.button("San Juan de Miraflores Income Data 2022", key="button_col1"):
        st.session_state.button_state_1 = not st.session_state.button_state_1
        st.session_state.button_state_2 = False  # Reset the state for other buttons
        st.session_state.button_state_3 = False
    if st.session_state.button_state_1:
        st.dataframe(df1[["PAYMENT_DATE", "MONTH", "CONCEPT", "2022", "2021", "2020", "2019", "2018", "TOTAL"]])

    if button_col2.button("Pucusana Income Data 2022", key="button_col2"):
        st.session_state.button_state_2 = not st.session_state.button_state_2
        st.session_state.button_state_1 = False  # Reset the state for other buttons
        st.session_state.button_state_3 = False
    if st.session_state.button_state_2:
        st.dataframe(df2[['OBSERVATIONS', 'AMOUNT_C', 'RIGHT_C', 'INTEREST_C', 'READJUSTMENT_C',
                          'DISCOUNT_C', 'TOTAL_COLLECTED', 'RECEIVED', 'RETURN', 'CANCEL_DATE',
                          'CANCEL_DATE_1', 'PAYMENT_TYPE', 'YEAR_RECEIPT', 'ACCOUNT_CODE',
                          'TAX_CODE', 'QUOTA_NUMBER', 'PROPERTY_CODE', 'CONCEPT_D', 'AMOUNT_D',
                          'RIGHT_D', 'INTEREST_D', 'READJUSTMENT_D', 'DISCOUNT_D']])

    if button_col3.button("Santa Anita Income Data 2023", key="button_col3"):
        st.session_state.button_state_3 = not st.session_state.button_state_3
        st.session_state.button_state_1 = False
        st.session_state.button_state_2 = False  # Reset the state for other buttons
    if st.session_state.button_state_3:
        st.dataframe(df3[['MOVE_TYPE', 'OBSERVATIONS', 'AMOUNT_C', 'RIGHT/LAW', 'INTERESTS_C',
                          'READJUSTMENT_C', 'DISCOUNT_C', 'TOTAL_COLLECTED', 'RECEIVED',
                          'RETURNED', 'CANCEL_DATE', 'PAYMENT_TYPE', 'ORDER_NUMBER',
                          'YEAR_RECEIVED', 'RECEIPT_NUMBER', 'ACCOUNT_CODE', 'TAX_CODE',
                          'QUOTA_NUMBER', 'PROPERTY_CODE', 'CONCEPT', 'AMOUNT_D', 'RIGHT_D',
                          'INTEREST_D', 'READJUSTMENT_D', 'DISCOUNT_D', 'SUBTOTAL_D']])


    else:
        tab1, tab2, tab3 = st.tabs(["San Juan de Miraflores", "Pucusana", "Santa Anita"])

        with tab1:
            # 1. Total collected across different concepts (Top Revenue-generating Concepts)
            top_concepts = df1.groupby('CONCEPT')['TOTAL'].sum().nlargest(10).reset_index()
            fig_top_concepts = px.bar(top_concepts, x='CONCEPT', y='TOTAL',title="Top Revenue-generating Concepts")
            st.subheader("Total Income by Concept")
            st.plotly_chart(fig_top_concepts)
            st.write("The most common payment concepts are PROPERTY TAX, CLEANING, FEES, PARKS AND GARDENS, MUNICIPAL SECURITY, and "
                     "ADMINISTRATIVE FINE. These six payment concepts together account for over 80% of total revenue. "
                     "The remaining payment concepts are less common and contribute less to total revenue.")

            # 2: Monthly Trends
            df_resample = df1.groupby('MONTH')['TOTAL'].sum().reset_index()
            fig_monthly_trends = px.line(
                df_resample,
                x='MONTH',
                y='TOTAL',
                markers=True,
                labels={'TOTAL': 'Payment Amount (Peruvian Sol)', 'MONTH': 'Months(2022)'},
                title="Monthly Trends in Income Collection",
            )
            st.subheader("Monthly Trends in Income Collection")
            st.plotly_chart(fig_monthly_trends)
            st.write("There appears to be some seasonality in the data, with noticeable fluctuations in payment amounts throughout different months.")

            # 3.Payment Amounts for Specific Concepts Over Time
            # Select specific payment concepts for analysis
            selected_concepts = ['IMPUESTO PREDIAL', 'LIMPIEZA', 'SERENAZGO']
            df_selected_concepts = df1[df1['CONCEPT'].isin(selected_concepts)]
            df_selected_concepts_pivot = df_selected_concepts.pivot_table(values='TOTAL', index='MONTH',
                                                                          columns='CONCEPT',
                                                                          aggfunc='sum').reset_index()
            # Plot the time series of payment amounts for specific concepts
            fig_selected_concepts = px.line(
                df_selected_concepts_pivot,
                x='MONTH',
                y=selected_concepts,
                markers=True,
                labels={'value': 'Payment Amount (Peruvian Sol)', 'MONTH': 'Time (Months)'})
            fig_selected_concepts.update_layout(legend_title_text='Concept')
            st.subheader("Payment Amounts for Specific Concepts Over Time")
            st.plotly_chart(fig_selected_concepts)
            st.write("All concepts show increasing trends, with property tax having the highest overall amount and security the lowest."
                     "Property tax exhibits the most significant fluctuations, while cleaning payments are relatively stable.")


        with tab2:
            # 1. Total collected across different concepts (Top Revenue-generating Concepts)
            top_concepts_2 = df2.groupby('CONCEPT_D')['TOTAL_COLLECTED'].sum().nlargest(5).reset_index()
            fig_top_concepts_2 = px.bar(top_concepts_2, x='CONCEPT_D', y='TOTAL_COLLECTED', title="Top Revenue-generating Concepts")
            st.subheader("Total Income by Concept")
            st.plotly_chart(fig_top_concepts_2)
            st.write("Public cleaning is the most important source of revenue for the municipality, generating almost half of the total revenue from the top five concepts. "
                     "The other four concepts Security, Property Tax, Parks and Gardens are all relatively close in terms of revenue generated.")

            # 2. Total collected across different observations (Top Revenue-generating observations)
            top_obs = df2.groupby('OBSERVATIONS')['TOTAL_COLLECTED'].sum().nlargest(5).reset_index()
            fig_top_obs = px.bar(top_obs, x='OBSERVATIONS', y='TOTAL_COLLECTED',
                                        title="Revenue Distribution for Top Observations")
            st.plotly_chart(fig_top_obs)
            st.write("Property Tax and Municipal Taxes stand out as the highest observations for total collected amounts. Municipal Taxes is the second-highest observation, albeit relatively lower than the top observation. "
                     "(Property Tax, Arbitrators), and (Property Tax, Municipal Taxes, Tax Penalties) are also notable observations.")

            # 3. Tax concepts are more prevalent among the residents
            tax_concept_counts = df2['CONCEPT_D'].value_counts().head(5)
            fig_top_tax = px.bar(tax_concept_counts, x=tax_concept_counts.index, y=tax_concept_counts,
                                 title="Distribution of Tax Concepts")
            fig_top_obs.update_layout(xaxis_title='Tax Concept', yaxis_title='Number of Occurrences')
            st.plotly_chart(fig_top_tax)

        with tab3:
            # 1. Top revenue generating concepts
            st.subheader("Revenue Distribution")
            top_concepts_3 = df3.groupby('CONCEPT')['TOTAL_COLLECTED'].sum().nlargest(10).reset_index()
            fig_top_concepts_3 = px.bar(top_concepts_3, x='CONCEPT', y='TOTAL_COLLECTED', labels={'TOTAL_COLLECTED': 'Total Collected Amount (Peruvian Sol)',
                                      'CONCEPT': 'Concept'}, title="Revenue Across Top Concepts")
            st.plotly_chart(fig_top_concepts_3)
            st.write("The top revenue-generating concepts include taxes on non-sporting public spectacles, medical care, garbage collection, parks and gardens, street sweeping, Municipal Security, property tax, administrative control fines, subdivision-related payments, and public cleaning. "
                                 "These concepts indicate a diverse range of revenue sources, with fines and property-related taxes playing a significant role.")
            
            # top_10_concepts = df3.groupby('CONCEPT')['TOTAL_COLLECTED'].sum().nlargest(10).sort_values(ascending=False)
            # df_top_10 = df3[df3['CONCEPT'].isin(top_10_concepts.index)]
            # fig_top_concepts = px.bar(
            #     df_top_10,
            #     x='TOTAL_COLLECTED',
            #     y='CONCEPT',
            #     labels={'TOTAL_COLLECTED': 'Total Collected Amount (Peruvian Sol)', 'CONCEPT': 'Concept'},
            #     title="Top 10 Revenue Generating Concepts",
            #     orientation='h',
            #     color='TOTAL_COLLECTED'
            # )
            # st.plotly_chart(fig_top_concepts)
            
            # 2. Distribution for Top Observations
            top_observations = df3.groupby('OBSERVATIONS')['TOTAL_COLLECTED'].sum().sort_values().head(10)
            df_top_observations = df3[df3['OBSERVATIONS'].isin(top_observations.index)]
            fig_top_observations = px.bar(
                df_top_observations,
                x='TOTAL_COLLECTED',
                y='OBSERVATIONS',
                labels={'TOTAL_COLLECTED': 'Total Collected Amount (Peruvian Sol)', 'OBSERVATIONS': 'Observations'},
                title="Revenue Distribution for Top Observations",
                orientation='h',  # Horizontal bar chart
                color='TOTAL_COLLECTED'  # Color bars based on the total collected amount
            )
            st.plotly_chart(fig_top_observations)
            st.write("Observations such as 'simple copies', 'additional receipts', and specific references like 'pool universal 19' have minimal impact on revenue, each contributing less than 2% to the total collected amount.")
            
            # 3. Total Revenue Collected by Payment Method
            total_collected_per_payment_method = df3.groupby('PAYMENT_TYPE')['TOTAL_COLLECTED'].sum().sort_values(ascending=False).reset_index()
            fig_payment = px.bar(total_collected_per_payment_method, x='PAYMENT_TYPE', y='TOTAL_COLLECTED',
                                 title='Total Revenue Collected by Payment Method',
                                 labels={'TOTAL_COLLECTED': 'Total Collected Amount (Peruvian Sol)',
                                         'PAYMENT_TYPE': 'Payment Method'})
            st.plotly_chart(fig_payment)
            st.write("The municipality primarily receives revenue through 'TRANSFERS', 'CASH', and credit card payments ('VISA'). While cash transactions are numerous, transfers constitute the highest total revenue.")
            
            # 4. Monthly Trends in Revenue Collection
            df3['CANCEL_DATE'] = pd.to_datetime(df3['CANCEL_DATE'])
            monthly_revenue = df3.groupby(df3['CANCEL_DATE'].dt.to_period("M"))['TOTAL_COLLECTED'].sum().reset_index()
            monthly_revenue['CANCEL_DATE'] = monthly_revenue['CANCEL_DATE'].astype(str)
            fig_monthly_trends = px.line(
                monthly_revenue,
                x='CANCEL_DATE',
                y='TOTAL_COLLECTED',
                markers=True,
                labels={'TOTAL_COLLECTED': 'Payment Amount (Peruvian Sol)', 'CANCEL_DATE': 'Months(2023)'}
            )
            st.subheader("Monthly Trends in Income Collection")
            st.plotly_chart(fig_monthly_trends)
            st.write("Revenue collection shows variations across months, with notable increases in February, April, and May. The municipality collected the highest revenue in April, contributing significantly to the overall trend.")
            
            
             
