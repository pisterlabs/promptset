from flask import Flask, render_template, request, jsonify
import pandas as pd
import yfinance as yf
import json
import numpy as np
from datetime import datetime
import openai
import plotly
import plotly.graph_objects as go
import charts.balance_chat as bc
import charts.candle_chart as cc
import trading_logics.mean_reversion as mr_strategy

# Flask app initialization
app = Flask(__name__)
# app.config["TEMPLATES_AUTO_RELOAD"] = True

# Define the root route with a form for input
@app.route('/')
def index():
    return render_template('index.html')  # Assuming you have an index.html file with the form


# Endpoint to handle form submission and simulate trading
@app.route('/simulate_trading', methods=['POST'])
def simulate_trading():
    # Fetch and process data (add your own data processing function)
    data = yf.download(tickers="BTC-USD", period='1y', interval='1d')
    
    # Simulate trading (add your own trading simulation function)
    asset_df, trade_actions, final_balance = mr_strategy.simulate_mean_reversion_trading(data, 10000, 0.025, 0.025, lookback_window=20)
    balance_df = pd.DataFrame(asset_df)
    
    # Create charts using Plotly (adapt your plotting functions)
    balance_chart = bc.plot_balance_chart(balance_df, trade_actions)
    candlestick_chart = cc.plot_candlestick_chart(data, trade_actions)
    
    # Convert charts to JSON for web rendering
    balance_chart_json = json.dumps(balance_chart, cls=plotly.utils.PlotlyJSONEncoder)
    candlestick_chart_json = json.dumps(candlestick_chart, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render results template with charts and final balance information
    return jsonify({
        'balance_chart': balance_chart_json,
        'candlestick_chart': candlestick_chart_json,
        'final_balance': final_balance
    })