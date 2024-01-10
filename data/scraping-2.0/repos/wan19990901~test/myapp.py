import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

countbyyear= pd.read_csv('countbyyear.csv')
bird_list = [{'label': y, 'value': y} for x, y in zip(countbyyear['species_id'], countbyyear['species_name'])]
state_list = [{'label': x, 'value': y} for x, y in zip(countbyyear['iso_subdivision'], countbyyear['iso_subdivision'])]
def drop_duplicate_dicts(list_of_dicts):
    unique_dicts = set()
    result = []

    for d in list_of_dicts:
        # Convert the dictionary to a JSON string
        json_str = json.dumps(d, sort_keys=True)

        # Check if this JSON representation has already been added
        if json_str not in unique_dicts:
            unique_dicts.add(json_str)
            result.append(d)

    return result

bird_list = drop_duplicate_dicts(bird_list)
state_list = drop_duplicate_dicts(state_list)

iso_subdivisions = countbyyear['iso_subdivision'].unique()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

mymarkdown = '''
This is my final project on bird banding dataset, I hope you can find the birds you are interested in and know about where to see them. Here are my data sources.

* [North American Bird Banding Program](https://www.sciencebase.gov/catalog/)
'''

countbyyear[countbyyear.species_name == 'Blue Goose']

countbyyear.head()

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# specification of what goes on
app.layout = html.Div(
    [
        # Stuff on Top
        html.H1("See where the birds were and where they are going!"),
        html.H2("This is a final project of University of Virginia Data Engineering Course."),
        html.H3("M.Y."),
        # Side Bar
        html.Div([
            dcc.Markdown('Please select the bird you like!'),
            dcc.Dropdown(id='species_name', options=bird_list, value='Lesser Snow Goose'),
            # New Dropdown for States
            dcc.Markdown(mymarkdown)
        ], style={'width': '24%', 'float': 'left'}),
        ###### Main bar for birds###
        html.Div([
            dcc.Tabs([
                dcc.Tab(label = 'Bird Image', children = [
                    html.Div([html.Img(id = 'birdimage', style={'height':'100%', 'width':'100%'})], style = {'width': '24%', 'float':'left'}),
                    html.Div([dcc.Graph(id = 'heatmap')], style = {'width': '74%', 'float':'right'})
                 ]),

                dcc.Tab(label='Population Over Time', value='tab1', children=[
                    dcc.Checklist(
                        id='state-checklist',
                        options=[{'label': i, 'value': i} for i in iso_subdivisions],
                        value=list(iso_subdivisions),
                        labelStyle={'display': 'block'}  # Select all states by default
                    ),
                    dcc.Graph(id='population-over-time')]),

                dcc.Tab(label = 'Birds Distributed Over States', children = [
                    dcc.Graph(id = 'timeslider', style = {'height': '100%', 'width': '100%'})
            ])
            ])
        ], style = {'width': '74%', 'float': 'right'})
    ]
)


@app.callback(
    Output('state-checklist', 'options'),
    [Input('species_name', 'value')]
)
def update_state_checklist(selected_species):
    # You may want to replace this with logic to fetch states based on the selected species
    available_states = countbyyear[countbyyear['species_name'] == selected_species]['iso_subdivision'].unique()
    return [{'label': state, 'value': state} for state in iso_subdivisions]

# Callback to update the line plot based on selected states
@app.callback(
    Output('population-over-time', 'figure'),
    [Input('species_name', 'value'),
     Input('state-checklist', 'value')]
)
def update_population_plot(selected_species, selected_states):
    data = countbyyear[countbyyear.species_name == selected_species]
    filtered_data = data[(data['species_name'] == selected_species) & (data['iso_subdivision'].isin(selected_states))]
    fig = px.line(filtered_data, x='event_year', y='item_count', color='iso_subdivision', title=f'Population of {selected_species} Over Time')
    return fig

#########################################################################
@app.callback([Output(component_id = 'birdimage', component_property = 'src')],
             [Input(component_id = 'species_name', component_property = 'value')])

def birdimage(b):
    OPENAI_API_KEY=openaikey
    client = OpenAI(api_key = OPENAI_API_KEY)
    response = client.images.generate(
      model="dall-e-3",
      prompt=f"a realistic photo of {b} with no text",
      size="1024x1024",
      quality="standard",
      n=1,
    )
    image_url = response.data[0].url

    return [image_url]

#########################################################################

#########################################################################
@app.callback([Output(component_id = 'timeslider', component_property = 'figure')],
             [Input(component_id = 'species_name', component_property = 'value')])

def timeslider(b):

    df_sel = countbyyear[countbyyear.species_name == b]
    #df =df[df.species_name == b]
    # Create a list of all years

    years = sorted(df_sel['event_year'].unique())
    # Creating the figure
    fig = go.Figure()

    # Add one trace for each year
    for year in years:
        df_year = df_sel[df_sel['event_year'] == year]
        fig.add_trace(
            go.Choropleth(
                locations=df_year['iso_subdivision'],
                z=df_year['item_count'],
                text=df_year['species_name'],
                colorscale='Viridis',
                autocolorscale=False,
                showscale=True,
                #geojson = 'us-states.json',
                geojson='https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json',
                featureidkey="properties.name"
            )
        )

    # Make all traces invisible at the start
    for trace in fig.data:
        trace.visible = False

    # Make the first trace visible
    fig.data[0].visible = True

    # Create and add slider
    steps = []
    for i, year in enumerate(years):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(years)},
                  {"title": f"Item count for year: {year}"}],
            label=str(year)
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Year: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title='Birds population changes over time',
        height=600,
        width=1000   )

    return [fig]

#########################################################################
@app.callback([Output(component_id = 'heatmap', component_property = 'figure')],
             [Input(component_id = 'species_name', component_property = 'value')])

def heatmap(b):
    myquery = pd.read_csv('heatmap.csv')
    forheat = myquery[myquery.species_name == b]
    target_time = 2022
    target_df = forheat[forheat['event_year'] == target_time]
    if not target_df.empty:
        # Create a Plotly Figure
        target_df['hover_text'] = target_df.apply(lambda row: f"Geo: {row['iso_subdivision']}, Lat: {row['lat_dd']}, Lon: {row['lon_dd']}", axis=1)
        fig = px.scatter_geo(target_df, lat='lat_dd', lon='lon_dd',
                hover_name='hover_text', size='item_count',
                projection='mercator', title=f'Heatmap of bird population at {target_time}')
        return [fig]
    else:
        # Handle empty DataFrame case
        return px.scatter_geo(title='No Data Available')


#########################################################################

if __name__ == "__main__":
    app.run_server(host = '0.0.0.0', port = '8050', debug = True)