import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import openai  # Make sure to install the openai library and set up your API key

# Load the data
url = "2014_world_gdp_with_codes.csv"
df = pd.read_csv(url)

# Initialize the Dash app
app = dash.Dash(__name__)

# OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY_HERE"

# Define the layout of the app
app.layout = html.Div([
    html.H1("World GDP Analysis"),

    # Dropdown for selecting the color scale (top-right corner)
    dcc.Dropdown(
        id="color-scale-dropdown",
        options=[
            {"label": "Plasma", "value": "Plasma"},
            {"label": "Viridis", "value": "Viridis"},
            {"label": "Inferno", "value": "Inferno"}
        ],
        value="Plasma",
        style={
            'width': '50%',
            'position': 'absolute',
            'top': '10px',
            'right': '10px'
        }
    ),

    dcc.Graph(id="choropleth-map"),

    # Input for chat with GPT-3
    dcc.Input(
        id="chat-input",
        type="text",
        placeholder="Ask a question on world GDP...",
        style={
            'width': '50%',
            'padding': '10px',
            'border': '2px solid #008CBA',
            'border-radius': '5px',
            'font-size': '16px',
            'margin-top': '10px'
        }
    ),

    html.Div(id="chat-output")
])


@app.callback(
    [dash.dependencies.Output("choropleth-map", "figure"),
     dash.dependencies.Output("chat-output", "children")],
    [dash.dependencies.Input("color-scale-dropdown", "value"),
     dash.dependencies.Input("chat-input", "value")]
)
def update_choropleth_map_and_chat(color_scale, chat_input):
    # Generate response from GPT-3
    chat_response = ""
    if chat_input:
        chat_response = openai.Completion.create(
            engine="davinci", prompt=chat_input, max_tokens=50
        ).choices[0].text

    # Update the choropleth map
    fig = px.choropleth(
        df,
        locations="CODE",
        color="GDP (BILLIONS)",
        hover_name="COUNTRY",
        title="World GDP Map",
        color_continuous_scale=color_scale,
        projection="natural earth"
    )

    return fig, chat_response


if __name__ == '__main__':
    app.run_server(debug=True)
