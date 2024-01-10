#content_format.py
import openai
import os

openai.api_key = os.getenv("MY_SECRET")

# Input support levels
support_levels = {
    'price': [],
    'context': []
}

# Input resistance levels
resistance_levels = {
    'price': [],
    'context': []
}

# Input content list
content_types = ["News Stories", "Podcast", "Tweets", "Books", "Not Gonna Make It Events"]

content_lists = []

def fetch_headlines(lists):
    # Initialize an empty string to store the prompt
    prompt = "Curated Content\n"

    # Loop over each content type
    for group in lists:
        # Add the content type and count to the prompt
        prompt += f"\n{group['name']}: {len(group['urls'])}\n"

        # Add the instruction for this content type to the prompt
        prompt += "Provide the article headline and source for each of the following URLs in the format: 'Article Headline | (URL Source)'\n"

        # Add each URL for this content type to the prompt
        for url in group['urls']:
            prompt += f"{url}\n"

        # Add a separator between content types
        prompt += "\n---\n"

    # Send the prompt to the GPT-3.5-turbo model
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    # Return the model's response
    return response['choices'][0]['message']['content'].strip().split('\n---\n')

def create_report_content(report_data, support_levels, resistance_levels):
  # Create a dictionary of metrics and their values for txt output
  metrics_dict = {}
  for metric in report_data:
    metrics_dict[metric] = report_data[metric].values[0]

  # Add support levels text
  support_txt = "Support Levels:\n"
  for i in range(len(support_levels['price'])):
    support_txt += f"Price: {support_levels['price'][i]}, Context: {support_levels['context'][i]}\n"

  # Add resistance levels text
  resistance_txt = "Resistance Levels:\n"
  for i in range(len(resistance_levels['price'])):
    resistance_txt += f"Price: {resistance_levels['price'][i]}, Context: {resistance_levels['context'][i]}\n"

  # Function to format metric output
  def metric_output(metric_name, metric_value, metric_change):
    formatted_value = f"{(metric_value):,}"
    return f"{metric_name} : {formatted_value} | 30 Day Change : {round(metric_change, 2)}%\n"

  # Report Format
  report_txt_output = f"""Bitcoin Market & On-chain Data Input:
    {metric_output('PriceUSD', metrics_dict['PriceUSD'], metrics_dict['PriceUSD_30d_change'])}
    {metric_output('Marketcap', metrics_dict['CapMrktCurUSD'], metrics_dict['CapMrktCurUSD_30d_change'])}

    Market Commentary - Section
    {support_txt}
    
    {resistance_txt}
    
    Market Sentiment - Section
{metric_output('VtyDayRet30d', metrics_dict['VtyDayRet30d'], metrics_dict['VtyDayRet30d_30d_change'])}
{metric_output('VtyDayRet180d', metrics_dict['VtyDayRet180d'], metrics_dict['VtyDayRet180d_30d_change'])}
{metric_output('Fear Greed Index', metrics_dict['fear_greed_index'], metrics_dict['fear_greed_index_30d_change'])}
{metric_output('200 Day Moving Average Multiple', metrics_dict['200_day_multiple'], metrics_dict['200_day_multiple_30d_change'])}
{metric_output('200 Week Moving Average', metrics_dict['200_week_ma_priceUSD'], metrics_dict['200_week_ma_priceUSD_30d_change'])}

    Network Health - Section
{metric_output('SplyCur', metrics_dict['SplyCur'], metrics_dict['SplyCur_30d_change'])}
{metric_output('HashRate', metrics_dict['7_day_ma_HashRate'], metrics_dict['7_day_ma_HashRate_30d_change'])}
{metric_output('AdrActCnt', metrics_dict['AdrActCnt'], metrics_dict['AdrActCnt_30d_change'])}
{metric_output('TxCnt', metrics_dict['TxCnt'], metrics_dict['TxCnt_30d_change'])}
{metric_output('TxTfrValAdjUSD', metrics_dict['TxTfrValAdjUSD'], metrics_dict['TxTfrValAdjUSD_30d_change'])}

    Valuation Models - Section
{metric_output('Thermocap Multiple', metrics_dict['thermocap_multiple'], metrics_dict['thermocap_multiple_30d_change'])}
{metric_output('8x Thermocap Multiple', metrics_dict['thermocap_multiple_8'], metrics_dict['thermocap_multiple_8_30d_change'])}
{metric_output('32x Thermocap Multiple', metrics_dict['thermocap_multiple_32'], metrics_dict['thermocap_multiple_32_30d_change'])}
{metric_output('MVRV Ratio', metrics_dict['mvrv_ratio'], metrics_dict['mvrv_ratio_30d_change'])}
{metric_output('CapRealUSD', metrics_dict['CapRealUSD'], metrics_dict['CapRealUSD_30d_change'])}
{metric_output('Realised Price', metrics_dict['realised_price'], metrics_dict['realised_price_30d_change'])}
{metric_output('3x Realised Marketcap Multiple', metrics_dict['realizedcap_multiple_3'], metrics_dict['realizedcap_multiple_3_30d_change'])}
    """
  return report_txt_output

def generate_newsletter(report_txt):
  print("Starting To Generate The Newsletter")
  prompt_template = """
Update the Bitcoin market newsletter by incorporating the following preprocessed data using the provided template. Make sure that the data is accurately formatted with financial data rounding to two decimal places for all numbers, except for percentage values. Generate a detailed and interpretive analysis based on the data provided.

Newsletter Template:

It is important to note that the price of Bitcoin is highly volatile and can fluctuate significantly in a short period of time. As a result, it is crucial for investors to monitor the market price and other related metrics to make informed investment decisions.

The current price of Bitcoin, which stands at {bitcoin_price}, represents the current value of a single bitcoin. The market capitalization, which stands at {market_cap}, represents the total value of all bitcoins in circulation. The {price_change_30_days} in the past 30 days in the price of Bitcoin and the market capitalization suggests a {price_change_description}, which can be influenced by various factors such as economic news, geopolitical events, and overall market sentiment.

The current support levels for Bitcoin are at {support_levels_section}, indicating that this price point is likely to offer support for any potential downward price movements. On the other hand, the resistance levels at {resistance_levels_section} indicate that these price points may pose a challenge for any upward price movements.

Support Levels:
{support_levels} ({support_level_descriptions})

Resistance Levels:
{Resistance_levels} ({Resistance_level_descriptions})

Volatility provides an understanding of the market stability and helps investors make decisions based on their risk tolerance. Additionally, market sentiment indicators give an insight into the overall sentiment of the market, which is influenced by various factors such as news events and market participant opinions. By combining these two metrics, investors can get a more complete overview of market conditions, allowing them to make informed investment choices.

Currently, the 30-day and 180-day volatility indices are at {30_day_volatility} and {180_day_volatility} respectively, showing a {volatility_change_description} in the short-term ({volatility_change} in 30-day volatility).

The Fear and Greed Index is a composite index that measures market sentiment by aggregating data from various sources and providing a score between 0 and 100. The current value of the Fear and Greed Index is {fear_and_greed_index_value}, which suggests that market sentiment is {market_sentiment_description}.

On-Chain Analysis
On-chain analysis provides a fundamental perspective on the health and activity of the Bitcoin network by examining various metrics that provide insight into the underlying health of the network, its level of adoption and usage, and its potential for future growth.

Starting with the supply, there are {current_supply} bitcoins currently in circulation with a finite limit of {total_supply_limit}. When it comes to the network's health, the hash rate, which is a measure of computational power, currently stands at {hash_rate}, with a {30_day_hash_rate_change}. In terms of adoption and usage, we are seeing a healthy number of daily active addresses at {active_addresses}, and the number of daily transactions {daily_transaction_change_description}, with a current count of {num_transactions} and a {30_day_transaction_change}. Furthermore, the daily total transaction value is {daily_transaction_value_change_description}, reaching {total_transaction_value} with a {30_day_transaction_value_change}.

Valuation Models
It's important to note that on-chain valuation models for Bitcoin are still in their early stages of development and should be used with caution. While they have shown promising results in the past, they are not a perfect indicator of future market performance.
  
Overvaluation Levels:
{32x_Thermocap_Multiple} - 32x Thermocap Multiple
{3x_Realised_Marketcap Multiple} - 3x Realized Price

Undervaluation Levels:
{8x_Thermocap_Multiple} - 8x Thermocap Multiple
{Realised_Price} - Realized Price

The Thermocap multiple measures the value of Bitcoin relative to the total miner revenue, providing insight into the asset's price premium with respect to total revenue received by miners. Currently, the Thermocap multiple stands at {thermocap_multiple}, with a {30_day_thermocap_multiple_change}.

The MVRV ratio, which measures the market value of Bitcoin relative to its realized value, is currently at {mvrv_ratio}. The realized capitalization, the total value of all Bitcoin in circulation at the price they last moved, is currently at {realized_capitalization}. The realized price for bitcoin, or the result of dividing the realized capitalization by the total coin supply, is currently {realized_price}, with a {30_day_realized_price_change}.

In conclusion, the value of Bitcoin is volatile, which is why it is crucial for those involved in trading or investing to keep track of various metrics that provide valuable insight into the current state of the bitcoin market. The current market scenario is {market_scenario_description}, with a {price_change_30_days} {price_change_description} in price over the past 30 days and a {hash_rate_change_description}, number of active addresses, and transactions, signaling {change_description]. While on-chain valuation models suggest that the market is {valuation_description}.
New Data:

"""
  response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a newsletter writer."},
            {"role": "user", "content": f"{prompt_template}\n\n{report_txt}"},
        ]
    )

  return response['choices'][0]['message']['content'].strip()