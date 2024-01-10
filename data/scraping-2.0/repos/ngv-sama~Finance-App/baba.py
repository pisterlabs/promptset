import streamlit as st
import requests
import pandas as pd
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

# Set Hugging Face API token
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# Backend functions for Banking Assistant
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt,
                     llm=HuggingFaceHub(repo_id="google/flan-t5-large",
                                        model_kwargs={"temperature": 0,
                                                      "max_length": 64}))

history = {'question': [], 'answers': []}

def LLm(question):
    answer = llm_chain.run(question)
    history['question'].append(question)
    history['answers'].append(answer)
    return history

# Streamlit App
st.title("Finance App")

# Dropdown menu for options
option = st.selectbox("Select Option:", ["Wealth Management", "Stock Prediction", "Chat Bot"])

if option == "Wealth Management":
    st.title("Wealth Management")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload your spending data (CSV):", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Display summary of expenditure
        st.write("### Expenditure Summary")

        # Find the business with the most transactions
        most_transactions = data['Location/Business'].value_counts().idxmax()

        # Find the average transaction amount
        avg_transaction_amount = data['Transaction Amount'].mean()

        st.write(f"The business with the most transactions is: {most_transactions}")
        st.write(f"The average transaction amount is: {avg_transaction_amount:.2f}")

        # Display pie chart of top spenders
        st.write("### Top 3 Spendings by Business")

        # Get top 3 spenders
        top_spenders = data.groupby('Location/Business')['Transaction Amount'].sum().nlargest(3)
        others = data.groupby('Location/Business')['Transaction Amount'].sum().nsmallest(len(data['Location/Business'].unique())-3)

        # Combine small categories into 'Others'
        top_spenders = top_spenders.append(pd.Series(others.sum(), index=['Others']))

        st.pie_chart(top_spenders)

        # Display time series graph of money spent over time
        st.write("### Money Spent Over Time")

        # Plot given data in blue
        st.line_chart(data.set_index('Date Time')['Transaction Amount'])

elif option == "Stock Prediction":
    st.title("Stock Prediction")

    # Text input for symbol
    symbol_input = st.text_input("Enter Stock Symbol (e.g., AAPL):")

    if st.button("Predict Stock Prices"):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol_input}&apikey=AJZ5QW0UH11140S5'
        r = requests.get(url)
        data = r.json()

        df = pd.DataFrame(data['Time Series (Daily)']).T
        df.index = pd.to_datetime(df.index)
        df['4. close'] = df['4. close'].astype(float)

        # Predict future stock prices using SARIMA model
        model = SARIMAX(df['4. close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)
        future_dates = pd.date_range(df.index[-1], periods=30)
        future_predictions = results.get_forecast(steps=30)
        forecast_data = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_predictions.predicted_mean.values})

        # Display the predicted stock prices
        st.write("### Predicted Stock Prices")
        st.write(forecast_data)

elif option == "Chat Bot":
    st.title("Bank Baba - Your Banking Assistant")

    # User input
    user_input = st.text_input("What is your question?:", "")

    # Process input and get response
    if st.button("Send"):
        if user_input:
            history = LLm(user_input)
            st.write(f"You: {user_input}")
            st.write(f"Bot: {history['answers'][-1]}")
