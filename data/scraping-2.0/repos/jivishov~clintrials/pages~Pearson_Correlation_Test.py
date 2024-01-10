import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
#import openai
import utils


# Streamlit App
st.title("Pearson Correlation for Clinical Trials")
st.markdown('<a href="https://drive.google.com/drive/folders/1Fo3vRuh0MMHw8iHipQk8jaWnEiErRZ8L?usp=drive_link" target="_blank">Download sample datasets/Nümunə verilənləri endirin</a>', unsafe_allow_html=True)
# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.write(df.head())
    
    # Column selection
    col1, col2 = st.multiselect("Select two columns for Pearson correlation", df.columns.tolist(), default=df.columns.tolist()[:2])
    
    if len(col1) == 0 or len(col2) == 0:
        st.write("### Error: Please select two columns for Pearson correlation.")
    else:
        # Pearson correlation calculation
        corr_coefficient, p_value = scipy.stats.pearsonr(df[col1], df[col2])
        
        # EDA plot
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.scatter(df[col1], df[col2])
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title(f"Scatter plot of {col1} and {col2}")
        plt.tight_layout(rect=[0, 0.2, 1, 1])  # Make space for the new legend
        st.pyplot(fig)
        
        # Results and interpretation
        st.write(f"Correlation coefficient: {corr_coefficient}")
        st.write(f"P-value: {p_value}")

        # GPT-4 interpretation
        with st.spinner("GPT-4 is analysing your results..."):
            gpt4_response = utils.GPT4_Interpretation(
                "Pearson correlation test",
                f"Correlation coefficient is {corr_coefficient}, and the p-value is {p_value}."
            )
        
        st.subheader("GPT-4's Interpretation:")
        st.write(f"{gpt4_response.choices[0].message.content}")
