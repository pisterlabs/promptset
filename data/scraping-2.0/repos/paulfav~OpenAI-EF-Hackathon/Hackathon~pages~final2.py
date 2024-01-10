import openai
import requests
import json
import re
import streamlit as st
from functions import Get_Portfolio, AlphaVantageNewsSentiment
api_key = "ENTER OPENAI CODE HERE"

# Your other imports, initializations and function definitions remain the same...

# Streamlit code
st.title("💬 Kairos® Investment Assistant Chatbot")

page = st.radio("Choose a page:", ["Chatbot", "Portfolio Simulation"])

if page == "Chatbot":
    def convert_to_days(value, unit):
        if unit == "day" or unit == "days":
            return int(value)
        elif unit == "year" or unit == "years":
            return int(value) * 365
        elif unit == "minute" or unit == "minutes":
            return int(value) / (24 * 60)
        else:
            return None

    def extract_info_from_message(message):
        info_dict = {}

        # Extract risk level
        risk_match = re.search(r"(low|high)", message.lower())
        if risk_match:
            if risk_match.group(1) == "low":
                info_dict["risk"] ="min_variance"
            else :
                info_dict["risk"] ="max_sharpe"

        

        # Extract horizon in days
        horizon_match = re.search(r"a\s+(\d+)\s*-*\s*(day[s]*|year[s]*|minute[s]*)\s+investment\s+horizon", message)
        if horizon_match:
            value = horizon_match.group(1)
            unit = horizon_match.group(2).lower()
            info_dict["horizon"] = convert_to_days(value, unit)
        # Extract amount of money
        money_match = re.search(r"([€\$]?)(\d+(?:,\d{3})*)", message)
        if money_match:
            currency_symbol = money_match.group(1)
            amount = int(money_match.group(2).replace(",", ""))
        
            # Convert euros to dollars if needed
            if currency_symbol == "€":
                amount *= 1.09  # Conversion rate from euros to dollars
        
            info_dict["money_to_invest"] = amount
        stock_match = re.findall(r"\b[A-Z]{2,5}\b", message)  # Assuming ticker symbols are uppercase and between 1-5 characters long
        if stock_match:
            info_dict["stocks"] = stock_match
        return info_dict


    openai.api_key = api_key

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }
    # Ask a couple questions on what field of companies the user likes at the beginning.
    # Initialize the conversation with the assistant's role and task description
    conversation = [{
        "role": "assistant",
        "content": "You are a investment assistant and will talk to a potential investor. I want you to ask questions about whether the user wants low risk or high risk, the horizon in day he expects of his investments, the amount of money in dollars he expects to invest and which firm fields the user likes and suggest appropriate stocks. While you do not have the 4 elements keep asking. Write <<Kairos!>> and recap the informations once you have every element. Do not forget <<Kairos!>> when you finish!"
    },
    {
            "role": "assistant",
            "content": f"Internal State: {dict()}"
        }
    ]

    # Function to add a new user message and get the assistant's reply
    def ask_question(conversation, user_message):
        conversation.append({"role": "user", "content": user_message})
        data = {
            "model": "gpt-4",
            "messages": conversation,
            "temperature": 0.9
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        )
        response_data = response.json()
        assistant_message = response_data['choices'][0]['message']['content']
        #conversation.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    # Streamlit code

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "You are an investment assistant and will talk to a potential investor. I want you to ask questions about whether the user wants low risk or high risk, the horizon in days they expect for their investments, the amount of money in dollars they expect to invest, and which firm fields the user likes and suggest appropriate stocks. While you do not have the 4 elements, keep asking. Write <<Kairos!>> and recap the information once you have every element."
            }
        ]

    for msg in st.session_state.messages:
        if "You are an investment assistant" not in msg["content"]:
            st.chat_message(msg["role"]).write(msg["content"])

    extracted_info = {}
    if prompt := st.chat_input():

        #st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        response_text = ask_question(st.session_state.messages, prompt)
        msg = {"role": "assistant", "content": response_text}
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg["content"])

        if "Kairos!" in response_text or "suggestions" in response_text:
            extracted_info = extract_info_from_message(response_text)
            st.session_state.page = "Portfolio Simulation"
            st.session_state.extracted_info = extracted_info

elif page == "Portfolio Simulation":
    if "extracted_info" in st.session_state:
        extracted_info = st.session_state.extracted_info

        col1, col2 = st.columns(2)

        with col1:
            st.title('Kairos Portfolio Optimization')

        with col2:
            st.image('logo.jpg', use_column_width=True)

        st.sidebar.header('User Input Parameters')

        try : 
            horizon = extracted_info["horizon"]
            st.sidebar.slider('Investiment Horizon (days)', 30, 365, horizon)
        except : 
            horizon = st.sidebar.slider('Investiment Horizon (days)', 30, 365, 90)

        try : 
            money_to_invest = extracted_info["money_to_invest"]
            st.sidebar.number_input('Money to Invest ($)', 1000, 100000, money_to_invest) 
        except : 
            money_to_invest = st.sidebar.number_input('Money to Invest ($)', 1000, 100000, 10000)

            stocks = st.sidebar.text_input('Stocks (comma separated)', 'AAPL,GOOGL,MSFT').split(',')

        method = st.sidebar.selectbox('Optimization Method', ['min_variance', 'max_sharpe'])


        user_dict = {
            'horizon': horizon,
            'stocks': stocks,
            'money_to_invest': money_to_invest,
            'method': method
        }

        try:
            portfolio = Get_Portfolio(user_dict)
            
            st.markdown(f"### Optimal Portfolio Allocation for method: **{method}**")
            st.write(portfolio.allocation)
            
            st.markdown(f"### Performance Metrics")
            st.write(f"**Expected returns:** {round(portfolio.expected_return*100, 2)}%, **Volatility:** {round(portfolio.expected_vol,2)*100}%, **Sharpe Ratio:** {round(portfolio.expected_sharpe,2)}")
            
            st.markdown("---")
            
            # Chart columns
            chart1, chart2 = st.columns(2)
            
            with chart1:
                st.markdown(f"### Stock Data Over The Horizon")
                st.line_chart(portfolio.strategy.data)

            with chart2:
                st.markdown(f"### Past Portfolio Performance")
                st.line_chart(portfolio.portfolio_value)
            
            st.markdown("---")

            st.markdown("### News Sentiment on Portfolio Stocks")
            news_sentiment = AlphaVantageNewsSentiment()
            sentiments = {}
            
            for stock in stocks:
                sentiment = news_sentiment.fetch_sentiment(stock)
                if sentiment:
                    sentiments[stock] = sentiment

            for stock, sentiment in sentiments.items():
                st.markdown(f"**Sentiment for {stock}:** {sentiment}")

        except Exception as e:
            st.write(f"An error occurred: {e}")

    else:
        st.write("Please go to the Chatbot page and answer the questions first.")
