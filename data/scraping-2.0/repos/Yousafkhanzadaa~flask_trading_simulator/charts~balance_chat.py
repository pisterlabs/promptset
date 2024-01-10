import openai
import plotly
import plotly.graph_objects as go

def plot_balance_chart(asset_df, trade_actions):
    balance_chart = go.Figure(data=go.Scatter(x=asset_df['Date'], y=asset_df['Balance'], mode='lines', name='Balance'))
    # Add trade action annotations on the balance chart
    for trade_action in trade_actions:
        trade_date = trade_action['date']
        # Check if the trade_date exists in the balance_df index
        if trade_date in asset_df.index:
            balance_chart.add_annotation(
                x=trade_date,
                y=asset_df.loc[trade_date, 'Balance'],
                text=f"{trade_action['action'].capitalize()} @ {trade_action['price']:.2f}",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40 if trade_action['action'] == 'buy' else 40,
                bgcolor="#00cc96" if trade_action['action'] == 'buy' else "#EF553B",
                arrowcolor="#00cc96" if trade_action['action'] == 'buy' else "#EF553B"
            )

    balance_chart.update_layout(
        title='Balance Over Time',
        xaxis_title='Date',
        yaxis_title='Balance',
        yaxis_tickprefix='$',
        template='plotly_dark'
    )
    return balance_chart

def plot_candlestick_chart(data, trade_actions):
    fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price')])
    for trade in trade_actions:
        color = 'green' if trade['action'] == 'buy' else 'red'
        fig.add_annotation(x=trade['date'], y=trade['price'], text=trade['action'].capitalize(), showarrow=True, arrowhead=1, arrowcolor=color, bgcolor=color)
    fig.update_layout(title='Price Chart with Trade Annotations', xaxis_title='Date', yaxis_title='Price in USD', template='plotly_dark')
    return fig