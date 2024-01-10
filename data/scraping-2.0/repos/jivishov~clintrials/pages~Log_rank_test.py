import streamlit as st
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
#import openai
import utils

st.title("Log Rank Test for Clinical Trials")

# # Sidebar for future menu options
# st.sidebar.title("Menu")
# st.sidebar.text("Future options will go here.")

st.markdown('<a href="https://drive.google.com/drive/folders/1Fo3vRuh0MMHw8iHipQk8jaWnEiErRZ8L?usp=drive_link" target="_blank">Download sample datasets/Nümunə verilənləri endirin</a>', unsafe_allow_html=True)
# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read and display data
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.write(df.head())

    # Select columns
    time_column = st.selectbox("Select the Time column", df.columns.tolist())
    event_column = st.selectbox("Select the Event column", df.columns.tolist())
    group_column = st.selectbox("Select the Group column", df.columns.tolist())

    # Error Checks
    error_message = None
    if df[time_column].dtype not in ['int64', 'float64']:
        error_message = "Time column should contain numerical values."
    if df[event_column].dtype not in ['int64', 'float64']:
        error_message = "Event column should contain numerical values."
    if df[group_column].dtype not in ['object', 'int64', 'float64']:
        error_message = "Group column should contain categorical or numerical values."
        
    if error_message:
        st.write(f"### Error: {error_message}")
    else:
        # Further Checks
        results = None  # Initialize to None
        if not all(group in df[group_column].values for group in ['A', 'B']):
            st.write("### Error: The data must contain both 'A' and 'B' groups.")
        elif len(df[df[group_column] == 'A']) < 2 or len(df[df[group_column] == 'B']) < 2:
            st.write("### Error: Both 'A' and 'B' groups must contain at least two observations.")
        elif not df[event_column].isin([0, 1]).all():
            st.write("### Error: Event column must contain only 0 or 1.")
        else:
            try:
                # Perform Log Rank Test
                T = df[time_column]
                E = df[event_column]
                groups = df[group_column]
                results = logrank_test(T[groups == 'A'], T[groups == 'B'], event_observed_A=E[groups == 'A'], event_observed_B=E[groups == 'B'])
            except Exception as e:
                st.write(f"### Error: An error occurred during the Log Rank Test: {e}")

        if results:
            # Display Results
            st.write("Log Rank Test Results:")
            st.write(f"P-value: {results.p_value}")
            st.write(f"Test statistic: {results.test_statistic}")

        # EDA: Kaplan-Meier Survival Curve
        st.write("### Kaplan-Meier Survival Curve")
        kmf = KaplanMeierFitter()
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size
        for name, grouped_df in df.groupby(group_column):
            kmf.fit(grouped_df[time_column], grouped_df[event_column], label=name)
            kmf.plot(ax=ax)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)  # Horizontal legend below the plot
        plt.tight_layout(rect=[0, 0.2, 1, 1])  # Make space for the new legend
        st.pyplot(fig)

    with st.spinner("GPT-4 is analysing your results..."):
        gpt4_response=utils.GPT4_Interpretation("Log Rank Test", 
                                                f"P-value={results.p_value}, Test statistic={results.test_statistic}")
   
    st.subheader("GPT-4's Interpretation:")
    st.write(f"{gpt4_response.choices[0].message.content}")
