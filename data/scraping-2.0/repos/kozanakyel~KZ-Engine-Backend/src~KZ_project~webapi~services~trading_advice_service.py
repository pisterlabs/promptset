from dotenv import load_dotenv
import os
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from langchain import OpenAI
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI

from KZ_project.Infrastructure.services.kayze_assistant_service.kayze_assistant import (
    KayzeAssistant,
)


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key


verbose = True
llm = ChatOpenAI(temperature=0.9, openai_api_key=openai_api_key)

conversation_stages = {
    "1": "Introduction: Begin the conversation with a polite greeting and a brief introduction about the company and its services.",
    "2": "Discover Preferences: Ask the client about their hobbies, interests or other personal information to provide a more personalized service.",
    "3": "Education Service Presentation: Provide more detailed information about the education services offered by the company.",
    "4": "AI Trading Service Presentation: Provide more detailed information about the AI trading services offered by the company.",
    "5": "Close: Ask if they want to proceed with the service. This could be starting a trial, setting up a meeting, or any other suitable next step.",
    "6": "Company Info: Provide general information about company like what is company and what are purposes and aimed etc.",
    "7": "Trading Advice Service Presentation: Provide and give detailed trading advice about to asked specific coin or asset",
}

config = dict(
    agent_name="KayZe",
    agent_role="Service Representative",
    company_name="KZEngine",
    company_values="Our vision is helping people trading decision when buy and sell decion process, via the Artificial Intelligence and MAchine Learning process.",
    conversation_purpose="Choosing the right service for the client and showing them the best option.",
    conversation_history=[],
    conversation_type="talking",
    conversation_stage=conversation_stages.get(
        "1",
        "Introduction: Begin the conversation with a polite greeting and a brief introduction about the company and its services.",
    ),
)

kayze_agent = KayzeAssistant.from_llm(llm, verbose=False, **config)
kayze_agent.seed_agent()


def create_openai_model(model_name: str = "text-davinci-003", temperature: float = 0.7):
    openai = OpenAI(
        # model_name='text-davinci-003',
        model_name=model_name,
        temperature=temperature,
    )
    return openai


def create_fewshot_template():
    examples = [
        {
            "query": f"RSI indicator value is 70.34, MFI indicator value is 59.02, DMP indicator value is 31.3,"
            f" DMN indicator value is 12.77 and ADX indicator value is 41.26."
            f" What is your advice for trading for those indicator values?",
            "answer": "The RSI indicator value being above 70 at this moment indicates an overbought zone has been entered."
            "The MFI value trending above the average confirms the flow of money. On the DMi side, "
            "the DMP (positive directional movement indicator) value is above "
            "the DMN (negative directional movement indicator) value, "
            "and ADX is strongly trending above 25 and at 40 levels, "
            "indicating a strong bull trend that has entered in a short period of time. "
            "When considering the flow of money and the overbought zone, it may be advisable "
            "to take some profits and waiting next market movements.",
        },
        {
            "query": f"RSI indicator value is 40.14, MFI indicator value is 41, DMP indicator value is 21.01,"
            f" DMN indicator value is 23.67 and ADX indicator value is 20.76."
            f" What is your advice for trading for those indicator values?",
            "answer": "The RSI indicator value dropping around 40 indicates approaching the selling zone. "
            "The MFI index also dropping around 40 supports this. Although ADX suggests that there is no strong trend below 25, "
            "it can be observed that DMN is above DMP, creating selling pressure. "
            "My recommendation would be to wait for a better buying opportunity at this point.",
        },
    ]

    # create a example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"], template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """Answer the question based on the context below.
        You are the trading advisor. Also you are expert on RSI, MFI and DMP, DMI indicators.

        Context: RSI indicator value range betwwen 0-100. RSI value 70 and above meaning that overbought area.
        So you should sell your coin. also value 80 is a extreme overbought and you act cautiously.
        RSI value 30 that meaning is overselling area. and value 20 is extreme overselling are.
        if the RSI value are 20-30s you should bought this coin. 30-70 range waiting or
        you can look other indicator results.
        MFI indicator value range betwen 0-100. MFI value 80 and above meaning that overbought area.
        So you should sell your coin. also value 90 is a extreme overbought and you should act cautiously.
        MFI value 20 that meaning is overselling area. and value 10 is extreme overselling are.
        if the MFI value are 10-20s you should bought this coin. 20-80 range waiting or
        you can look other indicator results.
        Else it shows overselling condition between 0-25.
        DMI indicator is a collection of indicators including DMP, DMI, and ADX. The Plus Direction Indicator DMP and
        Minus Direction Indicator DMI show the current price direction. When the DMP is above DMN,
        the current price momentum is up. When the DMN is above DMP, the current price momentum is down.
        ADX measures the strength of the trend, either up or down; a reading above 25 indicates a strong trend.
        Here are some examples:
        """
    # and the suffix our user input and output indicator
    suffix = """
        User: {query}
        AI: """

    # now create the few shot prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n",
    )

    return few_shot_prompt_template


def get_response_llm(model, fewshot_template, query: str):
    return model(fewshot_template.format(query=query))


def fetch_data(symbol: str, period: str, interval: str):
    # Fetch Bitcoin data from Yahoo Finance
    ohlc_data = yf.download(
        tickers=symbol, period=period, interval=interval, progress=False
    )
    return ohlc_data


def calculate_dmi_rsi_mfi(data):
    data.ta.adx(length=14, append=True)
    data.ta.rsi(length=14, append=True)
    data.ta.mfi(length=14, append=True)
    data = data.dropna(axis=0)
    return data


def create_query(indicator_data, symbol):
    rsi_14 = indicator_data.RSI_14.iloc[-1]
    mfi_14 = indicator_data.MFI_14.iloc[-1]
    dmp_14 = indicator_data.DMP_14.iloc[-1]
    dmn_14 = indicator_data.DMN_14.iloc[-1]
    adx_14 = indicator_data.ADX_14.iloc[-1]
    query = (
        f"For {symbol}: RSI indicator value is {rsi_14:.2f}, MFI indicator value is \
            {mfi_14:.2f}, DMP indicator value is {dmp_14:.2f},"
        f" DMN indicator value is {dmn_14:.2f} and ADX indicator value is {adx_14:.2f}."
        f" What is your advice for trading for those indicator values?"
    )
    return query


def get_ohlc_data(symbol: str):  # 'BTC-USD'
    df = fetch_data(symbol, "1mo", "1h")
    indicator_data = calculate_dmi_rsi_mfi(df)

    return indicator_data


if __name__ == "__main__":
    symbol = "BTC-USD"
    openai = create_openai_model()
    fewshot = create_fewshot_template()
    df = get_ohlc_data(symbol)
    query_test = create_query(df, symbol)
    advice_test = get_response_llm(openai, fewshot, query_test)
    print(advice_test)
