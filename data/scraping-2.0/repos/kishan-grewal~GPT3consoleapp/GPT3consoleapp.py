from asyncio.windows_events import NULL
from cgitb import text

def main():
    # imports
    import requests
    import os
    import openai
    from dotenv import load_dotenv
    load_dotenv("api-key.env")
    # get api key to use open api for generation of text and images
    openai.api_key = os.getenv("OPENAI_API_KEY")
    ow_api_key = os.getenv("OPENWEATHER_API_KEY")
    
    # define how images are created and returns the urls they are hosted on
    def generateImage(prompt):
     response = openai.Image.create(
      prompt=prompt,
      n=1, # maximum uses per minute is 50
      size="512x512"
     )
     image_url = response['data'][0]['url']
     return image_url
    
    # define how text is generated and returned
    def generateText(prompt):
     response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      temperature=0.5,
      max_tokens=512,
      top_p=0.5,
      best_of=5, # maximum uses per minute is 20
      frequency_penalty=0.5,
      presence_penalty=0.5
     )
     text_output = response["choices"][0]["text"]
     text_output = "".join([s for s in text_output.strip().splitlines(True) if s.strip("\r\n").strip()])
     return text_output
    


    # pull the weather
    session = requests.Session()
    city = "Slough"
    ow_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={ow_api_key}&units=metric"
    ow_data = session.get(ow_url).json()
    desc = ow_data.get('weather')[0].get('description')
    temp = ow_data.get('main').get('temp')
    feels_like = ow_data.get('main').get('feels_like')
    temp_min = ow_data.get('main').get('temp_min')
    temp_max = ow_data.get('main').get('temp_max')
    pressure = ow_data.get('main').get('pressure')
    humidity = ow_data.get('main').get('humidity')
    wind_speed = ow_data.get('wind').get('speed')
    wind_direction = ow_data.get('wind').get('deg')
    cloud_percentage = ow_data.get('clouds').get('all')

    # text to be used in webpage is pre-generated
    nice_text = generateText(prompt=f"Make a witty comment about the weather where the weather can be described as: {desc}")
    nice_image = generateImage(prompt=desc)

    # html file made within same script so that the generated text can be easily used
    cwd = os.getcwd()
    file_html = open(cwd + "\\templates\\index.html", "w")
    # string interpolation used below (f string) to pass variables to the html file
    file_html.write(f'''<html>
    <head>
    <title>GPT3consoleapp</title>
    </head> 
    <body>
    <h1>website1</h1>
    <p>{nice_text}</p> 
    <img src="{nice_image}" alt="generated image goes here">
    <p>Actual temperature outside: {temp}*C</p>
    <p>What it feels like: {feels_like}*C</p>
    <p>Temperature range: {temp_min}*C -> {temp_max}*C</p>
    <p>Air pressure outside: {int(pressure)/1013.25} times normal</p>
    <p>Humidity in the air: {humidity}% of maximum</p>
    <p>Speed of the wind: {wind_speed}m/s</p>
    <p>Direction it is going: {wind_direction}* clockwise from the South</p>
    <p>Percentage of the sky covered with clouds: {cloud_percentage}%</p>
    <p>{ow_data}</p>
    </body>
    </html>''')
    file_html.close()
    # creates website and runs html file
    from flask import Flask, render_template
    app = Flask("website1")
    @app.route('/')
    def home():
        return render_template('index.html')

    app.run()

main()