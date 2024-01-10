import streamlit as st
import yfinance as yf
from openai import OpenAI

client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

company_name = ""

def get_stock_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Current Price": info.get("currentPrice"),
        "Today's Opening Price": info.get("open"),
        "Previous Closing Price": info.get("previousClose"),
        "Today's High": info.get("dayHigh"),
        "Today's Low": info.get("dayLow"),
        "52 Week High": info.get("fiftyTwoWeekHigh"),
        "52 Week Low": info.get("fiftyTwoWeekLow"),
        "Dividend Yield": info.get("dividendYield"),
        "Market Cap": info.get("marketCap"),
        "Volume": info.get("volume"),
        "Average Volume": info.get("averageVolume"),
        "P/E Ratio": info.get("trailingPE"),
        "Forward P/E Ratio": info.get("forwardPE"),
        "Earnings Per Share (EPS)": info.get("trailingEps"),
        "Earnings Date": info.get("earningsDate"),
        "Dividend Rate": info.get("dividendRate"),
        "Beta": info.get("beta"),
        "Trailing Annual Dividend Rate": info.get("trailingAnnualDividendRate"),
        "Trailing Annual Dividend Yield": info.get("trailingAnnualDividendYield"),
        "Float Shares": info.get("floatShares")
    }

def display_stock_info(stock_info, exchange):
    st.subheader(f"Stock Information for {company_name} on {exchange}")
    for key, value in stock_info.items():
        if value is not None:
            st.text(f"{key} ({exchange}): {value}")

def main():
    global company_name
    st.title("Stock Information")
    company_name = st.text_input("Enter the name of the company:")
    ticker_nse = f"{company_name}.NS"
    ticker_bse = f"{company_name}.BO"

    if st.button("Get Stock Information"):
        try:
            stock_info_nse = get_stock_info(ticker_nse)
            if stock_info_nse["Current Price"] is not None:
                display_stock_info(stock_info_nse, "NSE")
                # Construct the dynamic prompt
                prompt = f"How has the stock been performing recently, considering its current price at {stock_info_nse['Current Price']}, today's opening price at {stock_info_nse['Today\'s Opening Price']}, and the previous closing price at {stock_info_nse['Previous Closing Price']}? Additionally, what are the 52-week high and low values ({stock_info_nse['52 Week High']} and {stock_info_nse['52 Week Low']}), and how do these relate to the current state of the stock? Provide insights into the dividend yield ({stock_info_nse['Dividend Yield']}), market cap ({stock_info_nse['Market Cap']}), and volume traded today ({stock_info_nse['Volume']}). How does the P/E ratio ({stock_info_nse['P/E Ratio']}) and forward P/E ratio ({stock_info_nse['Forward P/E Ratio']}) indicate the stock's valuation? Consider the earnings per share (EPS) at {stock_info_nse['Earnings Per Share (EPS)']} and the upcoming earnings date ({stock_info_nse['Earnings Date']}). Lastly, explore factors such as the dividend rate ({stock_info_nse['Dividend Rate']}), beta ({stock_info_nse['Beta']}), trailing annual dividend rate ({stock_info_nse['Trailing Annual Dividend Rate']}), trailing annual dividend yield ({stock_info_nse['Trailing Annual Dividend Yield']}), and float shares ({stock_info_nse['Float Shares']}). Based on these variables, what predictions can you make about the stock's future performance?"
            else:
                stock_info_bse = get_stock_info(ticker_bse)
                if stock_info_bse["Current Price"] is not None:
                    display_stock_info(stock_info_bse, "BSE")
                    # Construct the dynamic prompt
                    prompt = f"How has the stock been performing recently, considering its current price at {stock_info_bse['Current Price']}, today's opening price at {stock_info_bse['Today\'s Opening Price']}, and the previous closing price at {stock_info_bse['Previous Closing Price']}? Additionally, what are the 52-week high and low values ({stock_info_bse['52 Week High']} and {stock_info_bse['52 Week Low']}), and how do these relate to the current state of the stock? Provide insights into the dividend yield ({stock_info_bse['Dividend Yield']}), market cap ({stock_info_bse['Market Cap']}), and volume traded today ({stock_info_bse['Volume']}). How does the P/E ratio ({stock_info_bse['P/E Ratio']}) and forward P/E ratio ({stock_info_bse['Forward P/E Ratio']}) indicate the stock's valuation? Consider the earnings per share (EPS) at {stock_info_bse['Earnings Per Share (EPS)']} and the upcoming earnings date ({stock_info_bse['Earnings Date']}). Lastly, explore factors such as the dividend rate ({stock_info_bse['Dividend Rate']}), beta ({stock_info_bse['Beta']}), trailing annual dividend rate ({stock_info_bse['Trailing Annual Dividend Rate']}), trailing annual dividend yield ({stock_info_bse['Trailing Annual Dividend Yield']}), and float shares ({stock_info_bse['Float Shares']}). Based on these variables, what predictions can you make about the stock's future performance?"
                else:
                    st.write(f"Company {company_name} not found in NSE or BSE.")
                    return
        except Exception as e:
            st.write("An error occurred:", e)
            return

        st.subheader("Stock Performance Evaluation Using AI")

        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            ):
                full_response += (response.choices[0].delta.content or "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
