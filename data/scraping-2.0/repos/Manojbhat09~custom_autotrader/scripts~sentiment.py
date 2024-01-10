import openai
import requests
import time
import yfinance as yf  # Import the yfinance library for real-time data

# Initialize OpenAI API
openai.api_key = "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"

# List of 269 tickers
tickers = [
    "AAL", "TCEHY", "ANSS", "BA", "FULC",  # ... (all 269 tickers here)
]

def get_score_from_gpt4(ticker, real_time_data):
    prompt = f"""
    [Score] refers to a score in 0-10 if there is a hype today about that ticker or not. 
    It can be some happening or news sentiment that is likely to affect the stock price. 
    0 means less likely to have a drastic change and 10 meaning high likely due to some news happening around the ticker.
    
    Real-time data for {ticker}:
    - Current Price: {real_time_data['Close']}
    - Volume: {real_time_data['Volume']}
    - Change: {real_time_data['Change']}
    
    Tell me the [Score] for the stock ticker {ticker} based on the above real-time data.
    """
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=10
    )
    score = response['choices'][0]['text'].strip()
    return float(score)

def get_real_time_data(ticker):
    data = yf.Ticker(ticker)
    hist = data.history(period="1d")  # Get today's data
    latest_data = hist.iloc[-1]  # Get the latest data
    change = latest_data['Close'] - latest_data['Open']  # Calculate change
    return {
        'Close': latest_data['Close'],
        'Volume': latest_data['Volume'],
        'Change': change
    }

# # Main loop to check [Score] for each ticker in intervals of 5 minutes
# while True:
#     for i in range(0, len(tickers), 40):  # Batch of 40
#         batch = tickers[i:i+40]
#         for ticker in batch:
#             real_time_data = get_real_time_data(ticker)
#             score = get_score_from_gpt4(ticker, real_time_data)
#             if score > 6:
#                 print(f"{ticker} has a [Score] of {score}.")
#         time.sleep(300)  # Wait for 5 minutes before the next batch

def gpt4():
    prompt = f"""
    Browese and find out how to do math with integrals for stock data
    Share links,And whats your knowledge cut-off?
    """
    # response = openai.Completion.create(
    #     engine="GPT-4",
    #     prompt=prompt,
    #     max_tokens=2000
    # )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"{prompt}"}
        ]
    )
    import pdb; pdb.set_trace()
    score = response['choices'][0]['text'].strip()
    return score

print(gpt4())


