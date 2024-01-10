#!/usr/bin/env python
# coding: utf-8

##IMPORTING RELEVANT VARIABLES
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import streamlit as st
import openai
from datetime import date, timedelta
from streamlit_option_menu import option_menu
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
st.set_option('deprecation.showPyplotGlobalUse', False)

# Calculate the start and end dates
#end_date_ = date.today()
#start_date_ = end_date_ - timedelta(days=1)

# Format the dates as strings in "YYYY-MM-DD" format
#start_date_str_ = start_date_.strftime("%Y-%m-%d")
#end_date_str_ = end_date_.strftime("%Y-%m-%d")




# Set up OpenAI API credentials
openai.api_key = st.secrets["auth_key"]

#!/usr/bin/env python
# coding: utf-8



def upload_file():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file, sheet_name=0, parse_dates=True, header=1)
        #data = data.drop(columns=['Unnamed: 5', 'Unnamed: 6'])
        data = data.dropna()
        return data







def visualize_data(data):
    st.subheader("Data Visualization")
    columns = list(data.columns)
    plt.rcParams["figure.figsize"] = [18, 10]
    plt.rcParams["figure.autolayout"] = True
    selected_columns = st.multiselect("Select columns to visualize", columns)
    if len(selected_columns) > 0:
        chart_type = st.selectbox("Select chart type", ["Line Plot", "Bar Plot", "Scatter Plot"])

        if chart_type == "Bar Plot":
            for column in selected_columns:
                plt.bar(data.index, data[column], label=column)
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                st.pyplot()
        elif chart_type == "Line Plot":
            fig = px.line(data, x="Date", y=selected_columns)
            st.plotly_chart(fig)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(data, x="Date", y=selected_columns)
            st.plotly_chart(fig)

        # Perform time series forecasting on selected_columns
        for column in selected_columns:
            # Split the data into train and test sets
            train_data = data[column].iloc[:-15]
            test_data = data[column].iloc[-15:]

            # Define exogenous variables if available
            exog_train = None  # Modify this with your exogenous variables for the training set
            exog_test = None  # Modify this with your exogenous variables for the test set

            # Convert the index to a DatetimeIndex
            train_data.index = pd.to_datetime(train_data.index)
            test_data.index = pd.to_datetime(test_data.index)

            # Fit a SARIMA model
            model = SARIMAX(train_data, order=(0, 0, 0), seasonal_order=(1, 0, 0, 12), exog=exog_train)
            model_fit = model.fit()

            # Forecast future values
            forecast = model_fit.get_forecast(steps=15, exog=exog_test)

            # Extract the forecasted values and confidence intervals
            forecast_values = forecast.predicted_mean
            confidence_intervals = forecast.conf_int()

            # Convert confidence_intervals to DataFrame
            confidence_intervals_df = pd.DataFrame(confidence_intervals, index=test_data.index)

            # Plot the forecast
            plt.plot(test_data.index, test_data, label="Actual")
            plt.plot(test_data.index, forecast_values, label="Forecast")
            plt.fill_between(test_data.index, confidence_intervals_df.iloc[:, 0], confidence_intervals_df.iloc[:, 1], alpha=0.3)
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            st.pyplot()
# Function to handle user queries using ChatGPT
def handle_chatbot(query, data):
    # ChatGPT API call
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=query,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
        presence_penalty=0.2,
        frequency_penalty=0.0,
    )
    
    return response.choices[0].text.strip()



def main():
    st.set_page_config(page_title='Price Estimator')
    st.sidebar.title("Main Menu")
    selected = st.sidebar.selectbox("Select Option", ["Welcome", "Upload", "Estimator","Visualize",'Chatbot'])

    if selected == 'Welcome':
        st.write("# Welcome to Gidi's Price Estimator!👋")
        st.markdown("""This web app was developed by Gidi Richard to help estimate oil prices for a coming window 
                     given the price of the current window.""")

    elif selected == "Visualize":
        st.subheader('Visualize Data')
        data = upload_file()
        if data is None:
            st.warning('Please upload a file first.')
            return

        visualize_data(data)


    elif selected == 'Upload':
        st.subheader('Upload Data')
        data = upload_file()
        if data is not None:
            st.success('File uploaded successfully!')
           

    elif selected == 'Estimator':
        st.subheader('Price Estimator')

        data = upload_file()
        if data is None:
            st.warning('Please upload a file first.')
            return

    
        data['date'] = pd.to_datetime(data['Date'])

        st.subheader('OLD PRICING WINDOW')
        start_date = st.date_input(label='Starting Date', format="YYYY-MM-DD").strftime('%Y-%m-%d')
        end_date = st.date_input(label='End Date', format="YYYY-MM-DD").strftime('%Y-%m-%d')
        date_range = data.loc[(data['date'] >= start_date) & (data['date'] <= end_date)]
        df_columns = date_range[['Gasoline', 'Naphtha', 'Gasoil', 'LPG']]
        data1 = df_columns.mean()
        data1 = data1.reset_index()
        data1 = data1.rename(columns={'index': 'Product', 0: 'Average'})

        st.subheader('NEW PRICING WINDOW')
        start_date2 = st.date_input(label='New Starting Date', format="YYYY-MM-DD").strftime('%Y-%m-%d')
        end_date2 = st.date_input(label='New Ending Date', format="YYYY-MM-DD").strftime('%Y-%m-%d')
        date_range2 = data.loc[(data['date'] >= start_date2) & (data['date'] <= end_date2)]

        df_columns2 = date_range2[['Gasoline', 'Naphtha', 'Gasoil', 'LPG']]
        data2 = df_columns2.mean()
        data2 = data2.reset_index()
        data2 = data2.rename(columns={'index': 'Product', 0: 'New Average'})

        result = pd.concat([data1, data2], axis=1)
        new_data = result.loc[:, ~result.T.duplicated(keep='first')]

        new = new_data.T
        new = new.reset_index()
        new = new.drop('index', axis=1)
        new = new.rename(columns={1: 'Naphtha', 0: 'Gasoline', 2: 'Gasoil', 3: 'LPG'})
        new = new.drop(0)

        final = new.pct_change().dropna()

        st.subheader('CALCULATOR')
        product = st.selectbox('Select Product', options=final.columns)
        price = st.number_input(label='Current Price')

        calculate_conversion = st.checkbox('Calculate GHS/L conversion')
        if calculate_conversion:
            volume_gasoil = 1180
            volume_gasoline = 1300
            volume_naphtha = 1351.35
            volume_lpg = 1724.14

            volume = None
            if product == 'Gasoil':
                volume = volume_gasoil
            elif product == 'Gasoline':
                volume = volume_gasoline
            elif product == 'Naphtha':
                volume = volume_naphtha
            elif product == 'LPG':
                volume = volume_lpg
            else:
                volume = 1.0

            fx_rate = st.number_input(label='FX Rate')

            if fx_rate is not None and volume is not None:
                ghs_per_liter = ((new[product].values[1] + 80) / volume) * fx_rate
                st.write(f'The GHS/L conversion for {product} is {ghs_per_liter:.2f}')


        submit = st.button('Submit')

        if submit:
            percentage_change = final[product].values[0]
            if product == 'Gasoil':
                estimated_price = (percentage_change * price) + price
            else:
                estimated_price = (percentage_change * price) + price

            st.write(f'The estimated price of {product} is Ghc {estimated_price:.2f}')
            if percentage_change < 0:
                st.write(f'The price of {product} has reduced by a percentage of {percentage_change * 100:.2f}')
            else:
                st.write(f'The price of {product} has increased by a percentage of {percentage_change * 100:.2f}')


    elif selected == 'Chatbot':
        st.subheader('Chatbot')
        data = upload_file()
        if data is None:
            st.warning('Please upload a file first.')
            return

        query = st.text_input("Ask a question")
        if query:
            response = handle_chatbot(query, data)
            st.write("Chatbot: ", response)



if __name__ == '__main__':
    main()
