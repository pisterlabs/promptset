import openai
import image_download

# Openai key
openai.api_key = "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"

# Create model
response = openai.Completion.create(
  engine="davinci",
  prompt="Text: A car is a wheeled motor vehicle used for transportation. Cars are a primary means of transportation in many regions of the world. The year 1886 is regarded as the birth year of the car when German inventor Karl Benz patented his Benz Patent\n\nKeywords:",
  temperature=0.3,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.8,
  presence_penalty=0.0,
  stop=["\n"]
)

# Model Output
print(response.choices[0].text)

# convert enumeration into list
daten = response.choices[0].text
einzeldaten = daten.split(",")
print(einzeldaten)

# download images from google search
image_download.keywordPictures(einzeldaten)


