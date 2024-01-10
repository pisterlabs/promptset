import dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
from langchain.tools import tool
import streamlit as st

from tools.PolygonAPIClient import PolygonClient


dotenv.load_dotenv()
config = dotenv.dotenv_values()


@tool("get_technical_indicators", return_direct=True)
def get_technical_indicators(query):
    """
    Get technical indicators for a given symbol.
    """

    llm = ChatOpenAI(openai_api_key=config["OPENAI_API_KEY"],
                     streaming=False,
                     model_name="gpt-4",
                     temperature="0.5",
                     max_tokens="2048")

    with st.spinner("Internal Agent [Technical Indicators Agent] is transforming your request..."):
        json_request_url = llm(
            [ChatMessage(role="user", content=f"""
                        The user asked the following query to another GPT agent:
    
                        - {query}
    
                        ---
    
                        Based on the user's query, you need to query an API to provide the required technical indicators to
                        other agent. The agent might need this information to make a decision about a stock, or something
                        else. Still, your only task is to create the API request parameters for the other agent.
                        Your task is to generate the API request parameters with a "space character" between each parameter.
                        
                        The technical indicators are:
                        - SMA: Simple Moving Average
                        - EMA: Exponential Moving Average
                        - MACD: Moving Average Convergence Divergence
                        - RSI: Relative Strength Index
                        
                        These technical indicators are calculated in the function you will call, and you don't need to
                        calculate them yourself. You only need to provide the API request parameters to the other agent.
    
                        The API request parameters are:
                        - ticker_symbol : The symbol of the ticker in the financial / stocks market (e.g. AAPL)
                        - window : The window size used to calculate the SMA and EMA technical indicators. (e.g. 10)
                        - timespan : The period of time to get the technical indicators for, the options are:
                                - minute, hour, day, week, month, quarter, year
                        - macd_short_window : The short window size used to calculate the MACD technical indicator. (e.g. 12)
                        - macd_long_window : The long window size used to calculate the MACD technical indicator. (e.g. 26)
                        - macd_signal_window : The signal window size used to calculate the MACD technical indicator. (e.g. 9)
                        - rsi_window : The window size used to calculate the RSI technical indicator. (e.g. 14)
                        - series_type : The series type used to calculate the technical indicators, the options are:
                                - close, open, high, low
                        - is_adjusted : Whether the technical indicators are adjusted for stock splits or not. (e.g. true)
                        - max_limit : The maximum number of technical indicators to get. (e.g. 10)
                                Please not that the maximum limit is 32, and further requests will still return 32 technical
                                indicators.
    
                        ---
    
                        Here is an example of what you must return:
    
                        AAPL 10 day 12 26 9 14 close true 10
    
                        ---
                    """)]
        )
    st.success("Internal Agent [Technical Indicators Agent] has transformed your request successfully!")

    with st.spinner("Internal Agent [Technical Indicators Agent] is querying the API..."):
        parameters = json_request_url.content.split(" ")
        ticker_symbol = parameters[0].strip()
        window = parameters[1].strip()
        timespan = parameters[2].strip()
        macd_short_window = parameters[3].strip()
        macd_long_window = parameters[4].strip()
        macd_signal_window = parameters[5].strip()
        rsi_window = parameters[6].strip()
        series_type = parameters[7].strip()
        is_adjusted = parameters[8].strip()
        max_limit = parameters[9].strip()

        p = PolygonClient()
        indicators = p.get_technical_indicators(
            ticker_symbol=ticker_symbol,
            window=window,
            timespan=timespan,
            macd_short_window=macd_short_window,
            macd_long_window=macd_long_window,
            macd_signal_window=macd_signal_window,
            rsi_window=rsi_window,
            series_type=series_type,
            is_adjusted=is_adjusted,
            max_limit=max_limit
        )
    st.success("Internal Agent [Technical Indicators Agent] has queried the API successfully!")

    with st.expander("Reference Data [Technical Indicators API]", expanded=False):
        st.warning("\n\n" + str(indicators) + "\n\n")
        st.warning("Source: \n\n [1]  " + config["POLYGON_API_URL"] + "\n\n")

    return indicators


if __name__ == "__main__":
    print(get_technical_indicators("What are the last 30 days of prices for AAPL?"))
