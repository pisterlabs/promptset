import streamlit as st 
from lida import Manager, TextGenerationConfig , llm  
from dotenv import load_dotenv
import os
import openai
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import json
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

buffer = BytesIO()

@st.cache
def forecast():
    model = ARIMA(df['Value'], order=(1, 2, 3))
    results = model.fit()

    # future_points = st.text_area("Behold the Future", height=15)
    future_points = 30
    forecast = results.forecast(steps=int(future_points))
    forecast_index = pd.date_range(start=df.index[-1], periods=int(future_points)+1, freq='M')[1:]

    plt.figure(figsize=(12, 8))

    # Plot original data in blue
    plt.plot(df.index, df['Value'], label='Original Data', color='blue')

    # Plot forecasted data in red
    plt.plot(forecast_index, forecast, label='Forecasted Data', color='red')

    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Original Time Series Data and Forecast')
    plt.legend()

    plt.savefig(buffer, format='png')  # Save the plot as PNG in memory
    buffer.seek(0)  # Move the buffer cursor to the beginning

    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    img2 = base64_to_image(plot_base64)
    st.image(img2)

if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


lida = Manager(text_gen = llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)

menu = st.sidebar.selectbox("Choose an Option", ["Summarize", "Question based Graph"])

if menu == "Summarize":
    st.subheader("Summarization of your Data")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "filename.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        summary = lida.summarize("filename.csv", summary_method="default", textgen_config=textgen_config)
        st.write(summary)
        goals = lida.goals(summary, n=2, textgen_config=textgen_config)
        for goal in goals:
            st.write(goal)
        i = 0
        library = "seaborn"
        textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
        charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)  
        img_base64_string = charts[0].raster
        img = base64_to_image(img_base64_string)
        st.image(img)
        


        
elif menu == "Question based Graph":
    st.subheader("Query your Data to Generate Graph")
    file_uploader = st.file_uploader("Upload your CSV", type="json")
    if file_uploader is not None:
        print(file_uploader.file_id)
        path_to_save = "filename1.json"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        text_area = st.text_area("Query your Data to Generate Graph", height=200)
        with open(path_to_save, 'r') as file:
            data = json.load(file)

        df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])
        df.index = pd.to_datetime(df.index, unit='ms') 
        df.to_csv(f"{path_to_save[:-5]}.csv")
        if st.button("Generate Graph"):
            if len(text_area) > 0:
                st.info("Your Query: " + text_area)
                lida = Manager(text_gen = llm("openai")) 
                textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                summary = lida.summarize("filename1.csv", summary_method="default", textgen_config=textgen_config)
                user_query = text_area
                charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
                charts[0]
                image_base64 = charts[0].raster
                img = base64_to_image(image_base64)
                st.image(img)

                model = ARIMA(df['Value'], order=(1, 2, 3))
                results = model.fit()


                # future_points = st.text_area("Behold the Future", height=15)
                age = st.slider('How many Future Points to forecast?', 0, 130, 25)
                st.write("Providing forecast for next ", age, 'datapoints')

                # if st.button("Forecast", on_click=forecast):
                #     st.write("Generating forecast")
                # st.button("Forecast", on_click=clicked, args=[1])
                # if st.session_state.clicked[1]:
                #     if len(text_area) > 0:

                forecast = results.forecast(steps=int(age))
                forecast_index = pd.date_range(start=df.index[-1], periods=int(age)+1, freq='M')[1:]

                plt.figure(figsize=(12, 8))

                # Plot original data in blue
                plt.plot(df.index, df['Value'], label='Original Data', color='blue')

                # Plot forecasted data in red
                plt.plot(forecast_index, forecast, label='Forecasted Data', color='red')

                plt.xlabel('Date')
                plt.ylabel('Value')
                plt.title('Original Time Series Data and Forecast')
                plt.legend()

                plt.savefig(buffer, format='png')  # Save the plot as PNG in memory
                buffer.seek(0)  # Move the buffer cursor to the beginning

                plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                img2 = base64_to_image(plot_base64)
                st.image(img2)
        
