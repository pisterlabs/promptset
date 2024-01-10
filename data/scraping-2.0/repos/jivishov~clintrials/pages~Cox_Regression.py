import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
import openai
import utils


# Streamlit layout and logic
st.title("Cox Regression for Clinical Trials")

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

    # EDA: Display descriptive statistics
    st.write("### Exploratory Data Analysis")
    st.write("Descriptive statistics:")
    st.write(df.describe())

    # Select columns
    time_column = st.selectbox("Select the Time column", df.columns.tolist())
    event_column = st.selectbox("Select the Event column", df.columns.tolist())
    predictor_columns = st.multiselect("Select Predictor columns", df.columns.tolist(), default=df.columns.tolist())

    # Error checks
    error_message = None
    if time_column in predictor_columns:
        error_message = "Time column should not be selected as a Predictor column."
    if event_column in predictor_columns:
        error_message = "Event column should not be selected as a Predictor column."
    if df[predictor_columns].select_dtypes(include=['object']).any().any():
        error_message = "Predictor columns should only contain numerical values."
    
    if error_message:
        st.write(f"### Error: {error_message}")
    else:
        if time_column and event_column and predictor_columns:
            # Prepare data for Cox Regression
            selected_data = df[[time_column, event_column] + predictor_columns]
            
            # Fit Cox Regression model
            cph = CoxPHFitter()
            cph.fit(selected_data, duration_col=time_column, event_col=event_column)
            
            # Display results
            st.write("### Cox Regression Results")
            st.write(cph.summary)
            
            # EDA: Survival curves for each level of a categorical variable
            st.write("### Survival Curves")
            for predictor in predictor_columns:
                fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size
                cph.plot_partial_effects_on_outcome(covariates=predictor, values=df[predictor].unique(), cmap='coolwarm', ax=ax)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)  # Horizontal legend below the plot
                plt.tight_layout(rect=[0, 0.2, 1, 1])  # Make space for the new legend
                st.pyplot(fig)
                plt.close(fig)

            # # Interpretation
            # st.write("### Interpretation")
            # st.write("The p-values in the results table can be used to determine the significance of each predictor. A p-value < 0.05 typically indicates a significant predictor.")

            cox_results_text = cph.summary.to_csv()
            
            # GPT-4 interpretation
            with st.spinner("GPT-4 is analysing your results..."):
                gpt4_response = utils.GPT4_Interpretation(
                    "Cox regression",
                    f"Results of the Cox regression test:{cox_results_text}"
                )
            
            st.subheader("GPT-4's Interpretation:")
            st.write(f"{gpt4_response.choices[0].message.content}")
