import streamlit as st
import requests
import pandas as pd
import math
import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import numpy as np
import requests
from statistics import mean
from scipy import stats
from config import IEX_CLOUD_API_TOKEN
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit import pyplot as st_plt
import os
from config import Openai_api_key
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma



#Declaring all the global variables
global benchmark


st.title("🦜🔗 Financial Research Recommendation Algorithms - Level I to IV")
message =  "This project is intended for users with an intermediate knowledge of Finance"
st.warning(message)
st.image('images/finance.jpg')
def main():
    global stocks_all
    # Retrieve the tickers from the app state or initialize as an empty list
    if 'tickers' not in st.session_state:
        st.session_state.tickers = []

    # Create a form for the ticker input
    with st.form(key='ticker_form'):
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker_input = st.text_input("Enter a stock ticker:", key='ticker_input')
        with col2:
            add_ticker_button = st.form_submit_button(label="Add +")
            remove_ticker_button = st.form_submit_button(label="Remove -")

        # Check if the ticker is already in the list
        is_duplicate = ticker_input in st.session_state.tickers

        # Add the entered ticker to the list when the user clicks the "+" button
        if add_ticker_button and not is_duplicate:
            st.session_state.tickers.append(ticker_input)

        # Remove the last ticker from the list when the "-" button is clicked
        if remove_ticker_button:
            if st.session_state.tickers:
                st.session_state.tickers.pop()

    # Create a dropdown menu for selecting a benchmark
    with st.form(key='benchmark_form'):
        col1, col2 = st.columns([2, 2])
        with col1:
            benchmark = st.selectbox("Select One benchmark (More will be added soon):", ["SPY", "DJIA", "HSI", "UKX", "SX5E", "SHCOMP", "N225", "STI", "NSEASI", "DFMGI", "ADXGI", "TASI"])
        with col2:
            add_benchmark_button = st.form_submit_button("Add +")
            remove_benchmark_button = st.form_submit_button("Remove -")


    # Add or remove the selected benchmark from the list when the user clicks the corresponding button
    if add_benchmark_button:
        st.session_state.tickers.append(benchmark)
    if remove_benchmark_button and benchmark in st.session_state.tickers:
        st.session_state.tickers.remove(benchmark)

    # Display the current list of tickers and benchmarks
    st.markdown("**Current tickers and benchmarks:**")
    st.write(st.session_state.tickers)

    # Add the Start Date and End Date
    with st.form(key='start_end_dates'):
        st.header("Data Fetching using IEX cloud, Langchain LLM and Yahoo Finance APIs")
        st.warning("Read the documentation to understand what each platform does technically")
        portfolio_size = st.number_input("Enter the value of your portfolio in ($):")
        try:
            val = float(portfolio_size)
        except ValueError:
            print("That's not a number! \n Try again:")
            portfolio_size = input("Enter the value of your portfolio in ($):")
        option = st.radio(
                            'Please select the platform you would like to use;', (
                            'Level I : Stock & Reward visualizer', 'Level I : LangChain Annual Report summary','Level II : Equal-Weight Optimizer', 'Level III: Quantitative momentum Strategizer' , 'Level IV : Value Investing Strategizer'
                            )
                        )
        col1, col2 = st.columns([2, 2])
        with col1:
            start_date = st.date_input("Start date:")
        with col2:
            end_date = st.date_input("End Date:")
        if start_date and end_date and st.form_submit_button("Submit"):
            # Perform download on all stocks
            stocks_all = yf.download(st.session_state.tickers, start=start_date, end=end_date)
            if not stocks_all.empty:
                st.success("Data downloaded successfully! Analyst Can Now Investigate A Stock's perfomance against the selected benchmark in each desired platform")
                if option == 'Level I : Stock & Reward visualizer':
                    close_all = stocks_all.loc[:, "Close"].copy()
                    plt.figure(figsize=(15, 8))
                    for ticker in close_all.columns:
                        plt.plot(close_all.index, close_all[ticker], label=ticker)
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('Close Price', fontsize=12)
                    plt.title('Closing Prices($)', fontsize=14)
                    plt.grid(True)
                    plt.legend(loc='upper left')
                    plt.tight_layout()
                    fig = plt.gcf()  # Get the current figure
                    st.pyplot(fig)

                    # Use the normalized_close_all DataFrame as needed
                    st.header("Normalization for Benchmark Comparison [100 used]")
                    normalized_close_all = close_all.div(close_all.iloc[0]).mul(100)
                    plt.figure(figsize=(15, 8))
                    for ticker in normalized_close_all.columns:
                        plt.plot(normalized_close_all.index, normalized_close_all[ticker], label=ticker)
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('Closing price ($)', fontsize=12)
                    plt.title('Normalized Closing Prices($)', fontsize=14)
                    plt.grid(True)
                    plt.legend(loc='upper left')
                    plt.tight_layout()
                    normalized_figure = plt.gcf()
                    st.pyplot(normalized_figure)
                                
                    st.header("Annual Risk & Return / Year using Mean & Stdev")
                    return_all_close = close_all.pct_change().dropna()
                    summary_close_all = return_all_close.describe().T.loc[:, ["mean", "std"]]
                    summary_close_all["mean"] = summary_close_all["mean"] * 252
                    summary_close_all["std"] = summary_close_all["std"] * np.sqrt(252)
                    plt.figure(figsize=(12,8))
                    for i in summary_close_all.index:
                        plt.scatter(summary_close_all.loc[i, "std"] + 0.002, summary_close_all.loc[i, "mean"]+0.002, s=15)
                        plt.text(summary_close_all.loc[i, "std"] + 0.002, summary_close_all.loc[i, "mean"]+0.002, i)  # Add label using plt.text()
                    plt.xlabel('Annual Risk Promise(%)', fontsize=12)
                    plt.ylabel('Annual Return Promise(%)', fontsize=12)
                    plt.title('Annual Risk/Return Portfolio for Each Ticker', fontsize=14)
                    plt.grid(True)
                    plt.legend(loc='upper left')
                    plt.tight_layout()
                    normalized_figure_final = plt.gcf()
                    st.pyplot(normalized_figure_final)              
                    max_mean = summary_close_all["mean"].max()
                    max_std = summary_close_all["std"].max()
                    min_mean = summary_close_all["mean"].min()
                    min_std = summary_close_all["std"].min()
                    high_mean_rows = summary_close_all[(summary_close_all["mean"] == max_mean)]
                    high_std_rows = summary_close_all[(summary_close_all["std"] == max_std)]
                    high_mean_std_rows = summary_close_all[(summary_close_all["mean"] == max_mean) & (summary_close_all["std"] == max_std)]
                    low_everything = summary_close_all[(summary_close_all["mean"] == min_mean) & (summary_close_all["std"] == min_std)]

                    if not high_mean_rows.empty:
                        message = f"Our Graph depicts {high_mean_rows.index[0]} has higher Annual returns in this case. Check Risk Promise to decide Long/Short holding positions" 
                        st.success(message)
                    if not high_std_rows.empty:
                        message_warning = f"Furthermore, {high_std_rows.index[0]} has a higher Annual Risk Promise. Check Annual Return(s) to decide A short holding position or a Non-Purchase" 
                        st.warning(message_warning)
                    if not high_mean_std_rows.empty:
                        message_warning_two = f"Our Graph depicts {high_mean_std_rows.index[0]} has higher Annual returns with Higher Risk in this case.Recommended exclusively for short holding" 
                        st.warning(message_warning_two)
                    if not low_everything.empty and low_everything.index[0] not in benchmark:
                        message_warning_three = f"Our Graph depicts {low_everything.index[0]} has lower Annual returns with Higher Risk in this case. Not recommended for purchase"
                        st.warning(message_warning_three) 
                    
                    st.header("Volatility & Sensitivity Graphs Per stock in Return Margins (%) and in ($) For All Business Days till Aforementioned End Date")
                    #The moving average sections 
                    data_columns = close_all.columns[0:-1]
                    # Define a color palette with desired colors
                    color_palette = sns.color_palette("Set1", len(data_columns))
                    color_palette_two = sns.color_palette("Set2", len(data_columns))

                    # Create a line plot for each ticker with moving averages
                    for i, column in enumerate(data_columns):
                        fig1, ax = plt.subplots(figsize=(12, 8))
                        ax.set_title("Moving Average in Closing Price Change per Day in ($) In relation to Closing Price per day")
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Price in ($)")
                        closing_prices = close_all[column]
                        moving_average = close_all[column].rolling(window=10).mean()  # Adjust the window size as desired
                        ax.plot(closing_prices.index, closing_prices, color=color_palette_two[i], label=column)
                        ax.plot(moving_average.index, moving_average, color=color_palette[i], linestyle='--', label=f'{column} (Moving Average)')
                        plt.xticks(rotation=45, ha='right')
                        ax.legend()
                        st.pyplot(fig1)

                    # Extract the tickers (column names excluding the first column)
                    data_columns = return_all_close.columns[0:-1]

                    # Define a color palette with desired colors
                    color_palette = sns.color_palette("Set2", len(data_columns))

                    # Create a histogram for each ticker
                    for i, column in enumerate(data_columns):
                        fig, ax = plt.subplots(figsize=(12, 8))
                        ax.set_title("Volatility in Return Margins(%) Change")
                        ax.set_xlabel("Return Margins(%)")
                        ax.set_ylabel("Frequency of Return Change")
                        return_all_close.loc[:, column].plot(kind="hist", bins=100, ax=ax, color=color_palette[i], label=column)
                        ax.legend()
                        st.pyplot(fig)
                    

                    color_palette_zero = sns.color_palette("Set1", len(data_columns))
                    close_all_edited = close_all
                    new_tables = []
                    data_columns_one = close_all_edited.iloc[:, :-1]
                    for i, column in enumerate(data_columns_one):
                        new_table = close_all_edited[[column]].copy()
                        new_table = new_table.assign(Daily_Returns=np.log(new_table[column].div(new_table[column].shift(1))))
                        new_table.dropna(inplace = True)
                        new_table = new_table.assign(Cummulative_Returns = new_table.Daily_Returns.cumsum().apply(np.exp))
                        new_tables.append(new_table)
                    for i, table in enumerate(new_tables):
                        fig_cum, ax = plt.subplots(figsize=(12, 8))
                        ax.set_title("Cummulative Return For $1 Investment over Selected Period For the Ticker")
                        table["Cummulative_Returns"].plot(ax=ax, color=color_palette_zero[i], label=data_columns_one.columns[i])
                        ax.legend()
                        st.pyplot(fig_cum)
                    
                    #The covariance-ticker Graph graph
                    st.header("Correlation & Covariance Analysis Per stock & Against Benchmark Referencing Annual Returns(%)")
                    plt.figure(figsize = (12,8))
                    sns.set(font_scale = 1.4)
                    plt.title("Correlation Heatmap ")
                    sns.heatmap(return_all_close.corr(), cmap="Reds", annot=True, annot_kws={"size":15}, vmax=0.6)
                    correlation_graph = plt.gcf()
                    st.pyplot(correlation_graph)
                    plt.figure(figsize = (12,8))
                    sns.set(font_scale = 1.4)
                    plt.title("Correlation Heatmap ")
                    sns.heatmap(return_all_close.cov(), cmap="Greens", annot=True, annot_kws={"size":15}, vmax=0.6)
                    covariance_graph = plt.gcf()
                    st.pyplot(covariance_graph)
                if option == 'Level II : Equal-Weight Optimizer':
                    my_columns = ['Ticker', 'Price', 'Market Capitalization', 'Number Of Shares to Buy']
                    final_dataframe = pd.DataFrame(columns=my_columns)

                    symbol_groups = list(chunks(st.session_state.tickers, len(st.session_state.tickers)))
                    symbol_strings = []
                    for i in range(0, len(symbol_groups)):
                        symbol_strings.append(','.join(symbol_groups[i]))
                    final_dataframe = pd.DataFrame(columns=my_columns)
                    if benchmark == 'SPY' :
                        for symbol_string in symbol_strings:
                            batch_api_call_url = f'https://cloud.iexapis.com/stable/stock/market/batch/?types=quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
                            data = requests.get(batch_api_call_url).json()
                            for symbol in symbol_string.split(','):
                                final_dataframe = final_dataframe.append(
                                    pd.Series([symbol,
                                            data[symbol]['quote']['latestPrice'],
                                            data[symbol]['quote']['marketCap'],
                                            data[symbol]['quote']['latestTime']
                                            ],
                                            index=my_columns),
                                        ignore_index=True)
                    position_size = float(portfolio_size) / len(final_dataframe.index)
                    for i in range(0, len(final_dataframe['Ticker'])):
                        final_dataframe.loc[i, 'Number Of Shares to Buy'] = math.floor(position_size / final_dataframe['Price'][i])          
                    st.write(final_dataframe)

                if option == 'Level I : LangChain Annual Report summary':
                    from langchain.agents.agent_toolkits import (
                        create_vectorstore_agent,
                        VectorStoreToolkit,
                        VectorStoreInfo
                    )
                    os.environ['OPENAI_API_KEY'] = Openai_api_key

                    # Create instance of OpenAI LLM
                    llm = OpenAI(temperature=0.1, verbose=True)
                    embeddings = OpenAIEmbeddings()

                    st.success('This App allows you to summarize the financial health of a company after uploading its Annual document')

                    # Create a file uploader in Streamlit
                    uploaded_file = st.file_uploader("Upload The Annual Financial document (PDF)", type="pdf")

                    if uploaded_file is not None:
                        # Save the uploaded file to a temporary location
                        with open("uploaded_document.pdf", "wb") as file:
                            file.write(uploaded_file.read())

                        # Load the uploaded PDF document using PyPDFLoader
                        loader = PyPDFLoader("uploaded_document.pdf")

                        # Split pages from the PDF
                        pages = loader.load_and_split()

                        # Load documents into the vector database (ChromaDB)
                        store = Chroma.from_documents(pages, embeddings, collection_name='uploaded_document')

                        # Create vectorstore info object
                        vectorstore_info = VectorStoreInfo(
                            name="uploaded_document",
                            description="Uploaded financial document as a PDF",
                            vectorstore=store
                        )

                        # Convert the document store into a langchain toolkit
                        toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

                        # Add the toolkit to an end-to-end LC
                        agent_executor = create_vectorstore_agent(
                            llm=llm,
                            toolkit=toolkit,
                            verbose=True
                        )

                        # Create a text input box for the user
                        prompt = st.text_input('Input your prompt here')

                        # If the user hits enter
                        if prompt:
                            # Then pass the prompt to the LLM
                            response = agent_executor.run(prompt)
                            # ...and write it out to the screen
                            st.write(response)

                            # With a streamlit expander  
                            with st.expander('Document Similarity Search'):
                                # Find the relevant pages
                                search = store.similarity_search_with_score(prompt) 
                                # Write out the first 
                                st.write(search[0][0].page_content)

                        # Delete the temporary file
                        os.remove("uploaded_document.pdf")
                if option == 'Level III: Quantitative momentum Strategizer':
                    # Blueprints for Showcasing the Final Data Frames
                    symbol_groups = list(chunks(st.session_state.tickers, len(st.session_state.tickers)))
                    symbol_strings = []
                    for i in range(0, len(symbol_groups)):
                        symbol_strings.append(','.join(symbol_groups[i]))

                    hqm_columns = [
                            'Ticker', 
                            'Price', 
                            'Recommended Number of shares to Buy', 
                            'One-Year Price Return', 
                            'One-Year Return Percentile',
                            'Six-Month Price Return',
                            'Six-Month Return Percentile',
                            'Three-Month Price Return',
                            'Three-Month Return Percentile',
                            'One-Month Price Return',
                            'One-Month Return Percentile',
                            'HQM Score'
                            ]
                    hqm_dataframe = pd.DataFrame(columns = hqm_columns)
                    if benchmark == 'SPY' :
                        for symbol_string in symbol_strings:
                            batch_api_call_url = f'https://cloud.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
                            data = requests.get(batch_api_call_url).json()
                            for symbol in symbol_string.split(','):
                                hqm_dataframe = hqm_dataframe.append(
                                                                pd.Series([symbol, 
                                                                        data[symbol]['quote']['latestPrice'],
                                                                        'N/A',
                                                                        data[symbol]['stats']['year1ChangePercent'],
                                                                        'N/A',
                                                                        data[symbol]['stats']['month6ChangePercent'],
                                                                        'N/A',
                                                                        data[symbol]['stats']['month3ChangePercent'],
                                                                        'N/A',
                                                                        data[symbol]['stats']['month1ChangePercent'],
                                                                        'N/A',
                                                                        'N/A'
                                                                        ], 
                                                                        index = hqm_columns), 
                                                                ignore_index = True)
                                            
                    time_periods = [
                            'One-Year',
                            'Six-Month',
                            'Three-Month',
                            'One-Month'
                            ]

                    for row in hqm_dataframe.index:
                        for time_period in time_periods:
                            hqm_dataframe.loc[row, f'{time_period} Return Percentile'] = stats.percentileofscore(hqm_dataframe[f'{time_period} Price Return'], hqm_dataframe.loc[row, f'{time_period} Price Return'])/100

                    # Print each percentile score to make sure it was calculated properly
                    for time_period in time_periods:
                        print(hqm_dataframe[f'{time_period} Return Percentile'])
                    for row in hqm_dataframe.index:
                        momentum_percentiles = []
                        for time_period in time_periods:
                            momentum_percentiles.append(hqm_dataframe.loc[row, f'{time_period} Return Percentile'])
                        hqm_dataframe.loc[row, 'HQM Score'] = mean(momentum_percentiles)
                    hqm_dataframe.sort_values(by = 'HQM Score', ascending = False)
                    position_size = float(portfolio_size) / len(hqm_dataframe.index)
                    for i in range(0, len(hqm_dataframe['Ticker'])):
                        hqm_dataframe.loc[i, 'Recommended Number of shares to Buy'] = math.floor(position_size / hqm_dataframe['Price'][i])         
                    st.write(hqm_dataframe)

                if option == 'Level IV : Value Investing Strategizer':
                    if portfolio_size is not None:
                        # Blueprints for Showcasing the Final Data Frames
                        symbol_groups = list(chunks(st.session_state.tickers, len(st.session_state.tickers)))
                        symbol_strings = []
                        for i in range(0, len(symbol_groups)):
                            symbol_strings.append(','.join(symbol_groups[i]))

                        rv_columns = [
                        'Ticker',
                        'Price',
                        'Number of Shares to Buy', 
                        'Price-to-Earnings Ratio',
                        'PE Percentile',
                        'Price-to-Book Ratio',
                        'PB Percentile',
                        'Price-to-Sales Ratio',
                        'PS Percentile',
                        'EV/EBITDA',
                        'EV/EBITDA Percentile',
                        'EV/GP',
                        'EV/GP Percentile',
                        'RV Score'
                        ]

                        rv_dataframe = pd.DataFrame(columns = rv_columns)

                        for symbol_string in symbol_strings:
                            batch_api_call_url = f'https://cloud.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=quote,advanced-stats&token={IEX_CLOUD_API_TOKEN}'
                            data = requests.get(batch_api_call_url).json()
                            for symbol in symbol_string.split(','):
                                enterprise_value = data[symbol]['advanced-stats']['enterpriseValue']
                                ebitda = data[symbol]['advanced-stats']['EBITDA']
                                gross_profit = data[symbol]['advanced-stats']['grossProfit']
                                
                                try:
                                    ev_to_ebitda = enterprise_value/ebitda
                                except TypeError:
                                    ev_to_ebitda = np.NaN
                                
                                try:
                                    ev_to_gross_profit = enterprise_value/gross_profit
                                except TypeError:
                                    ev_to_gross_profit = np.NaN
                                    
                                rv_dataframe = rv_dataframe.append(
                                    pd.Series([
                                        symbol,
                                        data[symbol]['quote']['latestPrice'],
                                        'N/A',
                                        data[symbol]['quote']['peRatio'],
                                        'N/A',
                                        data[symbol]['advanced-stats']['priceToBook'],
                                        'N/A',
                                        data[symbol]['advanced-stats']['priceToSales'],
                                        'N/A',
                                        ev_to_ebitda,
                                        'N/A',
                                        ev_to_gross_profit,
                                        'N/A',
                                        'N/A'
                                ],
                                index = rv_columns),
                                    ignore_index = True
                                )
                        #Removing all the Null axes       
                        for column in ['Price-to-Earnings Ratio', 'Price-to-Book Ratio','Price-to-Sales Ratio',  'EV/EBITDA','EV/GP']:
                            rv_dataframe[column].fillna(rv_dataframe[column].mean(), inplace = True)
                        #rv_dataframe[rv_dataframe.isnull().any(axis=1)]
                        metrics = {
                            'Price-to-Earnings Ratio': 'PE Percentile',
                            'Price-to-Book Ratio':'PB Percentile',
                            'Price-to-Sales Ratio': 'PS Percentile',
                            'EV/EBITDA':'EV/EBITDA Percentile',
                            'EV/GP':'EV/GP Percentile'}
                        for row in rv_dataframe.index:
                            for metric in metrics.keys():
                                rv_dataframe.loc[row, metrics[metric]] = stats.percentileofscore(rv_dataframe[metric], rv_dataframe.loc[row, metric])/100

                        # Print each percentile score to make sure it was calculated properly
                        for metric in metrics.values():
                            print(rv_dataframe[metric])

                        for row in rv_dataframe.index:
                            value_percentiles = []
                            for metric in metrics.keys():
                                value_percentiles.append(rv_dataframe.loc[row, metrics[metric]])
                            rv_dataframe.loc[row, 'RV Score'] = mean(value_percentiles)
                        position_size = float(portfolio_size) / len(rv_dataframe.index)
                        for i in range(0, len(rv_dataframe['Ticker'])):
                            rv_dataframe.loc[i, 'Number of Shares to Buy'] = int(math.floor(position_size / rv_dataframe['Price'][i]))


                        if rv_dataframe is not None:
                            st.write(rv_dataframe)
                    else:
                        portfolio_size = st.number_input("Enter the value of your portfolio in ($):")
                        try:
                            val = float(portfolio_size)
                        except ValueError:
                            print("That's not a number! \n Try again:")
                            portfolio_size = input("Enter the value of your portfolio in ($):")

            else:
                st.error("Failed to download data. Try analyzing later")
            
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":
    main()