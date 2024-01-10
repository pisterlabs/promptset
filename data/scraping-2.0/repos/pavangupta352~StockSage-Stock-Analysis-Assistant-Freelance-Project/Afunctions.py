# Description: This file contains all the functions used in the main.py file.
import json
import numpy as np
import openai
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from scipy import stats
import datetime
from dateutil.parser import parse


# Function to get the stock price of a company
def getStockPrice(ticker, date=None, year=None):
    if date is not None:  # date is not None and year is not None:
        stock = yf.Ticker(ticker)  # ticker is the stock symbol
        # enddate is the next day of the date provided
        enddate = datetime.datetime.strptime(
            date, '%Y-%m-%d')+datetime.timedelta(days=1)
        # data is the stock price for the date provided
        data = stock.history(start=date, end=enddate.strftime('%Y-%m-%d'))
        if data.empty:  # if no data is available for the date provided
            return "No data available for the specified date range."
        else:  # if data is available for the date provided
            return str(data.iloc[0].Close)
    elif year is not None:  # if year is not None and date is None:
        start_year = year+'-01-01'  # start_year is the first day of the year provided
        end_year = year+'-12-31'  # end_year is the last day of the year provided
        return str(yf.Ticker(ticker).history(period='1y', start=start_year, end=end_year).iloc[-1].Close)
    else:  # if year is None and date is None:
        return str(yf.Ticker(ticker).history(period='1y').iloc[-1].Close)

#  Function to calculate the simple moving average (SMA) of a stock


def calculateSMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])

# Function to calculate the exponential moving average (EMA) of a stock


def calculateEMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

# Function to calculate the relative strength index (RSI) of a stock


def calculateRSI(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=14-1, adjust=False).mean()
    ema_down = down.ewm(com=14-1, adjust=False).mean()
    rs = ema_up/ema_down
    return str(100 - (100/(1+rs)).iloc[-1])

# Function to calculate the moving average convergence/divergence (MACD) of a stock


def calculateMACD(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()
    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal
    return f'{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}'

# Function to plot the stock price of a company


def plotStockPrice(ticker, start_date=None, end_date=None):
    if start_date is None:  # if start_date is None and end_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)
                      ).strftime('%Y-%m-%d')  # start_date is the date one year ago
    if end_date is None:  # if start_date is not None and end_date is None:
        end_date = datetime.datetime.now().strftime(
            '%Y-%m-%d')  # end_date is the current date

    data = yf.Ticker(ticker).history(start=start_date, end=end_date)

    # Determine the title based on the provided date information
    if start_date == (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d') and end_date == datetime.datetime.now().strftime('%Y-%m-%d'):
        title = f'{ticker} for the last year. If you want the graph for a specific year or date range, please provide that'
    elif start_date != (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d') and end_date != datetime.datetime.now().strftime('%Y-%m-%d'):
        title = f'{ticker} Stock Price from {start_date} to {end_date}'

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close)
    plt.title(title)  # Set the dynamically determined title
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()
    return "plotted sucessfully in stock.png"

# Function to calculate the rolling volatility of a stock


def rollingVolatility(ticker, start_date=None, end_date=None):
    if start_date is None:  # if start_date is None and end_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)
                      ).strftime('%Y-%m-%d')  # start_date is the date one year ago
    if end_date is None:  # if start_date is not None and end_date is None:
        end_date = datetime.datetime.now().strftime(
            '%Y-%m-%d')  # end_date is the current date
    # Fetch historical data
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)

    if data.empty:
        return "No data available for the specified date range."

    # Calculate rolling volatility
    rolling_volatility = data['Close'].rolling(window=20).std()

    if rolling_volatility.dropna().empty:
        return "Not enough data for rolling volatility calculation."

    # Create a plot of rolling volatility
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, rolling_volatility,
             label='Rolling Volatility (20-day)')
    plt.title(f'Rolling Volatility for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.savefig('rolling_volatility.png')
    plt.close()
    return str(rolling_volatility.dropna().iloc[-1])


def volatility(ticker, start_date=None, end_date=None):
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)
    returns = data['Close'].pct_change().dropna()
    volatility = returns.std()
    return str(volatility)


def relative_volatility(ticker1, ticker2, start_date=None, end_date=None):
    vol1 = float(volatility(ticker1, start_date, end_date))
    vol2 = float(volatility(ticker2, start_date, end_date))
    relative_vol = vol1 - vol2
    return str(relative_vol)


# Function to calculate the Sharpe ratio of a stock
def sharpeRatio(ticker, start_date, end_date, risk_free_rate):
    # Download historical data for the given date range
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)

    # Calculate daily returns
    returns = data['Close'].pct_change().dropna()

    # Calculate average daily return
    avg_return = returns.mean()

    # Calculate daily risk (standard deviation)
    risk = returns.std()

    # Calculate the Sharpe ratio
    sharpe = (avg_return - risk_free_rate) / risk

    return str(sharpe)

# Function to calculate the beta ratio of a stock


def betaRatio(ticker1, ticker2, start_date=None, end_date=None):
    if start_date is None:  # if start_date is None and end_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)
                      ).strftime('%Y-%m-%d')  # start_date is the date one year ago
    if end_date is None:  # if start_date is not None and end_date is None:
        end_date = datetime.datetime.now().strftime(
            '%Y-%m-%d')  # end_date is the current date
    # Download historical data for the given date range
    data1 = yf.Ticker(ticker1).history(start=start_date, end=end_date)
    data2 = yf.Ticker(ticker2).history(start=start_date, end=end_date)

    # Calculate daily returns for both assets
    returns1 = data1['Close'].pct_change().dropna()
    returns2 = data2['Close'].pct_change().dropna()

    # Calculate beta using linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        returns2, returns1)

    return str(slope)

# Function to calculate the value at risk (VaR) of a stock


def valueAtRisk(ticker, start_date, end_date, confidence_interval):
    # Download historical data for the given date range
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)

    # Calculate daily returns
    returns = data['Close'].pct_change().dropna()

    # Calculate the value at risk (VaR) using the confidence interval
    var = np.percentile(returns, (1 - confidence_interval) * 100)

    return str(var)

# Function to calculate the annualized performance of a stock


def annualizedPerformance(ticker, start_date=None, end_date=None):
    if start_date is None:  # if start_date is None and end_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)
                      ).strftime('%Y-%m-%d')  # start_date is the date one year ago
    if end_date is None:  # if start_date is not None and end_date is None:
        end_date = datetime.datetime.now().strftime(
            '%Y-%m-%d')  # end_date is the current date
    # Download historical data for the given date range
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)

    # Calculate daily returns
    returns = data['Close'].pct_change().dropna()

    # Calculate annualized performance
    annualized_return = (1 + returns.mean()) ** 252 - 1

    return str(annualized_return)

# Function to calculate the information ratio of a stock


def information_ratio(ticker1, ticker2, start_date=None, end_date=None):
    if start_date is None:  # if start_date is None and end_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)
                      ).strftime('%Y-%m-%d')  # start_date is the date one year ago
    if end_date is None:  # if start_date is not None and end_date is None:
        end_date = datetime.datetime.now().strftime(
            '%Y-%m-%d')  # end_date is the current date
    # Download historical data for the given date range
    data1 = yf.Ticker(ticker1).history(start=start_date, end=end_date)
    data2 = yf.Ticker(ticker2).history(start=start_date, end=end_date)

    # Calculate daily returns for both assets
    returns1 = data1['Close'].pct_change().dropna()
    returns2 = data2['Close'].pct_change().dropna()

    # Calculate the active return (excess return)
    active_return = returns1 - returns2

    # Calculate the tracking error (standard deviation of active return)
    tracking_error = active_return.std()

    if tracking_error == 0:
        return "Tracking error is zero, information ratio is undefined."

    # Calculate the information ratio
    information_ratio = active_return.mean() / tracking_error

    return str(information_ratio)

# Function to calculate the rolling correlation of a stock


def rolling_correlation(ticker1, ticker2, start_date=None, end_date=None):
    if start_date is None:  # if start_date is None and end_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)
                      ).strftime('%Y-%m-%d')  # start_date is the date one year ago
    if end_date is None:  # if start_date is not None and end_date is None:
        end_date = datetime.datetime.now().strftime(
            '%Y-%m-%d')  # end_date is the current date
    # Download historical data for the given date range
    data1 = yf.Ticker(ticker1).history(start=start_date, end=end_date)
    data2 = yf.Ticker(ticker2).history(start=start_date, end=end_date)

    # Calculate daily returns for both assets
    returns1 = data1['Close'].pct_change().dropna()
    returns2 = data2['Close'].pct_change().dropna()

    # Calculate rolling correlation
    rolling_corr = returns1.rolling(window=20).corr(returns2)

    if rolling_corr.dropna().empty:
        return "Not enough data for rolling correlation calculation."

    return str(rolling_corr.dropna().iloc[-1])

# Function to calculate the maximum drawdown of a stock


def maximum_drawdown(ticker, start_date=None, end_date=None):
    if start_date is None:  # if start_date is None and end_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)
                      ).strftime('%Y-%m-%d')  # start_date is the date one year ago
    if end_date is None:  # if start_date is not None and end_date is None:
        end_date = datetime.datetime.now().strftime(
            '%Y-%m-%d')  # end_date is the current date
    # Download historical data for the given date range
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)

    # Calculate daily returns
    returns = data['Close'].pct_change().dropna()

    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()

    # Calculate maximum drawdown
    peak = cumulative_returns.expanding(min_periods=1).max()
    trough = cumulative_returns.expanding(min_periods=1).min()
    drawdown = (peak - trough) / peak

    max_drawdown = drawdown.max()

    return str(max_drawdown)

# Function to calculate the omega ratio of a stock


def omega_ratio(ticker, start_date, end_date, threshold_return):
    # Download historical data for the given date range
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)

    # Calculate daily returns
    returns = data['Close'].pct_change().dropna()

    # Calculate the number of returns exceeding the threshold
    num_positive_returns = (returns > threshold_return).sum()
    num_negative_returns = (returns < threshold_return).sum()

    if num_negative_returns == 0:
        return "Omega ratio is undefined because there are no negative returns."

    omega_ratio = num_positive_returns / num_negative_returns

    return str(omega_ratio)

# Function to calculate the R-squared of a stock


def r_squared(ticker1, ticker2, start_date=None, end_date=None):
    if start_date is None:  # if start_date is None and end_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)
                      ).strftime('%Y-%m-%d')  # start_date is the date one year ago
    if end_date is None:  # if start_date is not None and end_date is None:
        end_date = datetime.datetime.now().strftime(
            '%Y-%m-%d')  # end_date is the current date
    # Download historical data for the given date range
    data1 = yf.Ticker(ticker1).history(start=start_date, end=end_date)
    data2 = yf.Ticker(ticker2).history(start=start_date, end=end_date)

    # Calculate daily returns for both assets
    returns1 = data1['Close'].pct_change().dropna()
    returns2 = data2['Close'].pct_change().dropna()

    # Calculate the benchmark's variance
    benchmark_variance = returns2.var()

    if benchmark_variance == 0:
        return "R-squared is undefined because benchmark variance is zero."

    # Calculate the covariance between the asset and the benchmark
    covar = np.cov(returns1, returns2)[0, 1]

    # Calculate the asset's variance
    asset_variance = returns1.var()

    # Calculate R-squared
    r_squared = (covar / benchmark_variance) ** 2

    return str(r_squared)


def relative_return(ticker, benchmark_ticker, start_date=None, end_date=None):
    if start_date is None:
        start_date = (datetime.datetime.now() -
                      datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # Fetch the historical data
    data_ticker = yf.Ticker(ticker).history(start=start_date, end=end_date)
    data_benchmark = yf.Ticker(benchmark_ticker).history(
        start=start_date, end=end_date)

    # Calculate the returns
    returns_ticker = data_ticker['Close'].pct_change().dropna()
    returns_benchmark = data_benchmark['Close'].pct_change().dropna()

    # Calculate the relative return
    relative_return = returns_ticker - returns_benchmark

    return str(relative_return.iloc[-1])


def compare_performance(tickers, start_date=None, end_date=None):
    performances = {}
    for ticker in tickers:
        annualized_perf = annualizedPerformance(ticker, start_date, end_date)
        performances[ticker] = float(annualized_perf)

    # Find the ticker with the best performance
    best_ticker = max(performances, key=performances.get)

    # Prepare the response
    response = f"{best_ticker} performed better than the others because its annualized performance was {performances[best_ticker]}."

    for ticker, performance in performances.items():
        if ticker != best_ticker:
            relative_perf = relative_return(
                best_ticker, ticker, start_date, end_date)
            response += f" The relative performance of {best_ticker} to {ticker} was {relative_perf}."

    return response


def compare_risk(tickers, start_date=None, end_date=None):
    risks = {}
    for ticker in tickers:
        var = valueAtRisk(ticker, start_date, end_date,
                          0.05)  # 5% confidence interval
        drawdown = maximum_drawdown(ticker, start_date, end_date)
        risks[ticker] = (float(var), float(drawdown))

    # Find the ticker with the highest risk
    riskiest_ticker = max(risks, key=lambda x: risks[x][0] + risks[x][1])

    # Prepare the response
    response = f"{riskiest_ticker} is riskier than the others because its Value at Risk was {risks[riskiest_ticker][0]} and its maximum drawdown was {risks[riskiest_ticker][1]}."

    return response


def compare_volatility(tickers, start_date=None, end_date=None):
    volatilities = {}
    for ticker in tickers:
        vol = volatility(ticker, start_date, end_date)
        volatilities[ticker] = float(vol)

    # Find the ticker with the highest volatility
    most_volatile_ticker = max(volatilities, key=volatilities.get)

    # Prepare the response
    response = f"{most_volatile_ticker} was more volatile than the others because its volatility was {volatilities[most_volatile_ticker]}."

    return response


def compare_relative_volatility(tickers, start_date=None, end_date=None):
    # This function assumes that the first ticker in the list is the reference
    reference_ticker = tickers[0]
    relative_vols = {}
    for ticker in tickers[1:]:
        rel_vol = relative_volatility(
            reference_ticker, ticker, start_date, end_date)
        relative_vols[ticker] = float(rel_vol)

    # Find the ticker with the highest relative volatility
    most_volatile_ticker = max(relative_vols, key=relative_vols.get)

    # Prepare the response
    response = f"{most_volatile_ticker} had higher relative volatility than {reference_ticker} because its relative volatility was {relative_vols[most_volatile_ticker]}."

    return response


def compare_rolling_volatility(tickers, start_date=None, end_date=None):
    rolling_vols = {}
    for ticker in tickers:
        roll_vol = rollingVolatility(ticker, start_date, end_date)
        rolling_vols[ticker] = float(roll_vol)

    # Find the ticker with the highest rolling volatility
    most_volatile_ticker = max(rolling_vols, key=rolling_vols.get)

    # Prepare the response
    response = f"{most_volatile_ticker} had higher rolling volatility than the others because its rolling volatility was {rolling_vols[most_volatile_ticker]}."

    return response
