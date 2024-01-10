from ipywidgets import widgets
import datetime
import sys
import numpy as np                
import pandas as pd               ## data processing, dataset file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt   ## data visualization & graphical plotting
import seaborn as sns             ## to visualize random distributions
import plotly.express as px       ## data visualization & graphical plotting
import plotly.graph_objects as go ## data visualization & graphical plotting
import plotly.subplots as sp      ## data visualization & graphical plotting
import requests
import matplotlib.pyplot as plt
import requests
import matplotlib.pyplot as plt
from IPython.display import display
import folium
from IPython.display import display, HTML
import openai
import json



sys.path.append('D:/google_hackathon/project/VoilaDisasterPredictor/model/earthquake')
from tsunamiUtils import getTsunamiLinearRegressionResult
tsunamiDf =pd.read_csv("D:\google_hackathon\earthquake_data.csv")
class DisasterPredictorApp(object):
    def __init__(self):
        self._runDate = None
        self._tsunamiPredictionWidget = widgets.Output()
        self._weatherVisualizerWidget = widgets.Output()
        self._tsunamiPredictionInputWidget = widgets.Output()
        self._tsunamiPredictionOutputWidget = widgets.Output()
        self._tsunamiInfoWidget = widgets.Output()
        self._newsArticlesWidget = widgets.Output()
        self._weatherPlotWidgetResult = widgets.Output()
        self._weatherPlotWidgetInput = widgets.Output()
        self._chatBotWidget = widgets.Output()
        self.weather_api_key = "c4fa56be9b07826adb1cd1b4b82e612b"
        openai.api_key = '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'

    def getWidget(self):
        ''' This function returns the main widget of the app.
            The main tab consists of:
            1. tsunami Prediction 
            2. Weather Visualizer
            3. News Articles
            4. Chat Bot
        '''
        inputWidget = self.initInputWidget()
        resultWidget = self.initResultWidget()
        titleWidget = widgets.HTML(value='<h1>Safety Net</h1>')
        mainWidget = widgets.VBox(
            children=[titleWidget, inputWidget, resultWidget])
        return mainWidget

    def initInputWidget(self):
        self._runDate = widgets.DatePicker(description='Run Date',
                                           value=datetime.date.today(),
                                           disabled=False,
                                           layout=widgets.Layout(width='50%')
                                           )

        runButton = widgets.Button(description='Run',
                                   disabled=False,
                                   icon='check',
                                   button_style='primary',
                                   width=widgets.Layout(width='40px')
                                   )
        inputWidget = widgets.HBox([self._runDate, runButton])
        return inputWidget

    def refreshTab(self):
        self._tsunamiPredictionWidget.children = [self._tsunamiPredictionInputWidget, self._tsunamiPredictionOutputWidget, self._tsunamiInfoWidget]
        self._weatherVisualizerWidget.children = [self._weatherPlotWidgetInput, self._weatherPlotWidgetResult]
        self._resultTab.children = [
            self._tsunamiPredictionWidget, self._weatherVisualizerWidget, self._newsArticlesWidget, self._chatBotWidget]

    def initResultWidget(self):
        self._resultTab = widgets.Tab()
        self._resultTab.children = [
            self._tsunamiPredictionWidget, self._weatherVisualizerWidget, self._newsArticlesWidget, self._chatBotWidget]
        self._resultTab.set_title(0, 'Tsunami Prediction')
        self._resultTab.set_title(1, 'Weather Visualizer')
        self._resultTab.set_title(2, 'News Articles')
        self._resultTab.set_title(3, 'Chat Bot')
        try:
            self._weatherVisualizer()
            self._earthqaukePrediction()
            self._newsArticles()
            self._chatBot()
        except Exception as e:
            print(e)
        return self._resultTab

    # Function to interact with the bot
    def chat_with_bot(self, user_input):
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=user_input,
            max_tokens=50,
            temperature=0.7,
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()

    # Function to handle user input and update bot response
    def handle_user_input(self):
        user_input = self.user_input_widget.value
        self.bot_response = self.chat_with_bot(user_input)
        self.bot_response_widget.value = f'<p><b>Bot:</b> {self.bot_response}</p>'
        self.user_input_widget.value = ''


    def _chatBot(self):
        # Create widgets for user input and bot response
        self.user_input_widget = widgets.Text(placeholder='Enter your message...', layout=widgets.Layout(width='auto'))
        self.bot_response_widget = widgets.HTML()
        # Assign the handle_user_input function to the on_submit event of the user input widget
        #self.user_input_widget.on_submit(lambda _: self.handle_user_input())
        self.bot_response_widget.value = '<p><b>Bot:</b> Hello, I am a chatbot. I am here to help you with your queries related to natural disasters. Unfortunately I am out of service right now :(</p>'
        # Display the widgets
        self._chatBotWidget =  widgets.VBox([self.user_input_widget, self.bot_response_widget])
        self.refreshTab()

    def _newsArticles(self):
        children = [widgets.HTML(value='<h2>News Articles</h2>')]
        children.append(widgets.HTML('The following are the latest news articles related to natural disasters'))
        api_key = 'ee2b669b82d149c1b14c716287554753'
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': 'natural disaster',
            'apiKey': api_key
        }

        response = requests.get(url, params=params)
        data = response.json()

        out = widgets.Output()
        with out:
            if response.status_code == 200:
                articles = data['articles']
                for i, article in enumerate(articles):
                    title = article['title']
                    url = article['url']
                    image_url = article['urlToImage']
                    if image_url:
                        html_content = f"<div><img src='{image_url}' style='width:100px;height:100px;margin-right:10px;'><h3>{title}</h3></div><a href='{url}' target='_blank'>Read more</a>"
                    else:
                        html_content = f"<div><h3>{title}</h3></div><a href='{url}' target='_blank'>Read more</a>"
                    display(HTML(html_content))
            else:
                out = widgets.HTML('Failed to fetch news articles')
        children.append(out)
        self._newsArticlesWidget = widgets.VBox(children=children)
        self.refreshTab()

    def updatetsunamiResults(self, button = None):
        result, accuracy = getTsunamiLinearRegressionResult(self.modelName.value,
             self.month.value, self.cdi.value, self.mmi.value, self.alert.value, self.sig.value, self.net.value, self.nst.value, self.dmin.value, self.gap.value, self.magType.value, self.depth.value, self.latitude.value, self.longitude.value)
        if result == 1:
            text = 'The earthquake will result in a Tsunami'
        else:
            text = 'The earthquake will not result in a Tsunami'
        children = [widgets.HTML(text)]
        bodytext  = 'The Accuracy score of this mode is ' + str(accuracy)
        children.append(widgets.HTML(bodytext))
        self._tsunamiPredictionOutputWidget = widgets.VBox(children=children)

        self.refreshTab()

    def _earthqaukePrediction(self):
        children = [widgets.HTML(value='<h2>Tsunami Prediction</h2>')]
        children.append(widgets.HTML('If you want to know whether an earthquake will result in a Tsunami, please enter the following information:'))
        self.modelName = widgets.Dropdown(
            options=['Logistic Regression', 'SVM', 'Naive Bayes'],
            value='Logistic Regression',
            description='Model',
            disabled=False,
        )

        self.month = widgets.Dropdown(
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            value=11,
            description='Month:',
            disabled=False,
        )
        self.latitude = widgets.FloatText(
            value=-9.7963,
            min=-90,
            max=90,
            step=0.000001,
            description='Latitude:',
            disabled=False
        )
        self.longitude = widgets.FloatText(
            value=159.596,
            min=-180,
            max=180,
            step=0.000001,
            description='Longitude:',
            disabled=False
        )

        tsunamiInputWidget = widgets.HBox([self.month, self.latitude, self.longitude])
        children.append(tsunamiInputWidget)
        self.cdi = widgets.FloatText(
            value=8,
            min=-2.0,
            max=15.0,
            step=0.1,
            description='Community Density Intensity (CDI):',
            style={'description_width': 'initial'},
            disabled=False
        )
        self.mmi = widgets.BoundedIntText(
            value=7,
            min=0,
            max=10,
            step=1,
            description='Modified Mercalli intensity scale:',
            style={'description_width': 'initial'},
            disabled=False
        )

        self.alert = widgets.Dropdown(
            options=['green', 'yellow', 'orange', 'red'],
            value='green',
            description='Alert:',
            disabled=False
        )
        self.net = widgets.Dropdown(
            options=['ak', 'at', 'ci', 'duputel', 'hv', 'nc', 'nn', 'official', 'pt',
       'us', 'uw'],
            value='us',
            description='Net:',
            disabled=False
        )
        self.nst = widgets.BoundedIntText(
            value=10,
            min=0,
            max=10,
            step=1,
            description='No. of seismic stations ',
            style={'description_width': 'initial'},
            disabled=False)

        self.dmin = widgets.FloatText(
            value=0.509,
            min=0.0000,
            max=12.0000,
            step=0.0001,
            description='Hori dist-epicenter to nearest station',
            disabled=False,
            style={'description_width': 'initial'}

        )
        self.gap = widgets.BoundedIntText(
            value=17,
            min=0,
            max=360,
            step=1,
            description='largest azimuthal gap (degree) ',
            style={'description_width': 'initial'},
            disabled=False)

        self.magType = widgets.Dropdown(
            options=['Mi', 'mb', 'md', 'ml', 'ms', 'mw', 'mwb', 'mwc', 'mww'],
            value='mww',
            description='mag type:',
            disabled=False,

        )
        self.depth = widgets.FloatText(
            value=14,
            min=0.000,
            max=1000.000,
            step=0.001,
            description='Depth',
            disabled=False

        )
        self.sig = widgets.BoundedIntText(
            value=768,
            min=0,
            max=1000,
            step=1,
            description='Significance',
            disabled=False

        )
        children.append(widgets.HBox([self.modelName, self.cdi]))
        children.append(widgets.HBox([self.mmi, self.alert]))
        children.append(widgets.HBox([self.sig, self.nst]))
        children.append(widgets.HBox([self.dmin, self.gap]))
        children.append(widgets.HBox([self.depth, self.magType]))
        children.append(self.net)
        #children.append(maginitude)
        self._tsunamiPredictionOutputWidget = widgets.HTML(
            value='Results to be displayed')
        runButton = widgets.Button(description='Run')
        children.append(runButton)
        runButton.on_click(self.updatetsunamiResults)
        self._tsunamiPredictionInputWidget = widgets.VBox(children=children)

        m = folium.Map(location=[0, 200], zoom_start=2)
        
        # Add a marker for each data point
        for _, row in tsunamiDf.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup=f"Magnitude: {row['magnitude']}",
            ).add_to(m)

        
        
        # fig = px.density_mapbox(tsunamiDf, lat='latitude', lon='longitude', z='magnitude', radius=6,
        #                     center=dict(lat=0, lon=200), zoom=0, mapbox_style="stamen-terrain")
        # fig.update_geos(fitbounds="locations")
        out = widgets.Output()
        with out:
            display(m)
        tsunamiDetails = widgets.HTML('<h3> Some details about Earthquakes and Tsunamis</h3>')
        #self._tsunamiInfoWidget = widgets.VBox(children=[tsunamiDetails, out])
        earthquakeHistoricText = widgets.HTML('<h3> Earthquakes in the past</h3>')
        self._tsunamiInfoWidget = widgets.VBox(children=[earthquakeHistoricText, out])
        self._tsunamiPredictionWidget = widgets.VBox(
            children=[self._tsunamiPredictionInputWidget, self._tsunamiPredictionOutputWidget, self._tsunamiInfoWidget])
        self.refreshTab()

    def plot_weather_forecast(self, button = None):
        # Make API call to retrieve weather data
        url = 'http://api.openweathermap.org/data/2.5/forecast'
        params = {
            'q': self.locations.value,
            'appid': self.weather_api_key,
            'units': 'metric'
        }
        response = requests.get(url, params=params)
        data = response.json()

        # Extract relevant information
        timestamps = []
        temperatures = []
        humidities = []
        # Add more lists for other weather factors like 'humidities', 'pressures', etc.

        for forecast in data['list']:
            timestamp = forecast['dt']
            temperature = forecast['main']['temp']
            humidity = forecast['main']['humidity']
            # Add more variables for other weather factors like 'pressure', 'wind_speed', etc.
            
            timestamps.append(timestamp)
            temperatures.append(temperature)
            humidities.append(humidity)
            # Append other weather factors to their respective lists.

        # Convert timestamps to human-readable format
        dates = [datetime.datetime.fromtimestamp(timestamp) for timestamp in timestamps]
        
        # Create traces for different weather factors
        temperature_trace = go.Scatter(x=dates, y=temperatures, name='Temperature (°C)')
        humidity_trace = go.Scatter(x=dates, y=humidities, name='Humidity (%)')
        # Add more traces for other weather factors.

        # Create the figure layout
        layout = go.Layout(
            title='Weather Forecast',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Value'),
        )

        # Add all the traces to the figure
        data = [temperature_trace, humidity_trace]
        # Add more traces to the 'data' list.

        # Create the figure
        fig = go.Figure(data=data, layout=layout)

        # Display the figure
        out = go.FigureWidget(fig)
        children = []
        children.append(widgets.HTML(self.locations.value))
        children.append(out)
        self._weatherPlotWidgetResult = widgets.VBox(children=children)
        self.refreshTab()

    def _weatherVisualizer(self):
        self.locations = widgets.Dropdown(
            options=['London, UK', 'New York, United States', 'Delhi, India', 'Mumbai, India', 'Paris, France', 'Tokyo, Japan', 'Sydney, Australia'],
            value='London, UK',
            description='Location',
            disabled=False
        )
        children = []
        children.append(self.locations)
        weatherPlotButton = widgets.Button(description='Weather Plot')
        children.append(weatherPlotButton)
        
        weatherPlotButton.on_click(self.plot_weather_forecast)
        self.plot_weather_forecast()
        self._weatherPlotWidgetInput = widgets.VBox(children=children)

        self._weatherVisualizerWidget = widgets.VBox(
            children=[self._weatherPlotWidgetInput, self._weatherPlotWidgetResult])

        self.refreshTab()
