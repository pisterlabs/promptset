import datetime
import os.path
from openai import OpenAI

from pathlib import Path
from base64 import b64decode
from io import BytesIO

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

now = datetime.datetime.utcnow().isoformat() + "Z"  # 'Z' indicates UTC time

def main():
  """Shows basic usage of the Google Calendar API.
  Prints the start and name of the next 10 events on the user's calendar.
  """
  creds = None
  prompt=""
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("/token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
          #"credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

 
  try:
        service = build("calendar", "v3", credentials=creds)

        # Fetch all accessible calendars
        calendars_result = service.calendarList().list().execute()
        calendars = calendars_result.get('items', [])

        if not calendars:
            print("No calendars found.")
            return

        events = []
        for calendar in calendars:
            calendar_id = calendar['id']
            #print(f"Getting the upcoming 10 events for calendar: {calendar['summary']}")

            events_result = (
                service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=now,
                    maxResults=3,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            events += events_result.get("items", [])
            

        if not events:
            print(f"No upcoming events found for calendar: {calendar['summary']}")
        else:
            #prompt+="Today is Nov 11. Imagine you are a master prompt maker for dalle. You specialise in creating images based on my calendar entries. You’re creative, hide allegories and details. Give me a prompt based on todays calendar. The image should be in a style of 19th century litograph or metal plate print as would be seen in an old book or newspaper. The events list is:"
            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                prompt+=start+" "+str(event["summary"])+", "
            prompt+=". Keep the image simple, the scene simple and prompt close to the style i want, black and white and simple enough to print in an old book. Keep the prompt short and focused on my near term most important commitments. Remove any personally identifiable information and do not mention dates. Use these words in the beginning of the prompt for the specific style a vintage engraving, caravaggesque, flickr, ultrafine detail, neoclassicism, no margins, full screen. Respond with the prompt only. "

  except HttpError as error:
        print(f"An error occurred: {error}")

  # Set your API key here

  # Define the prompt
  prompti = prompt
  #print("Sending %s to chatgpt" % prompti)
  client = OpenAI(api_key = '') # Add your own

  # ChatGPT the prompt
  completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
      {"role": "system", "content": "you are a master prompt maker for dalle. You specialise in creating 19th century metal litograph images in the style of a vintage engraving, caravaggesque, flickr, ultrafine detail, neoclassicism based on my calendar entries. You are creative, hide allegories and details. Give me a prompt based on todays calendar. The image should be in a style of 19th century litograph or metal plate print as would be seen in an old book or newspaper. Today is %s " %  now},
      {"role": "user", "content": prompti}
    ]
  )

  #print(completion.choices[0].message.content)

  # Generate the image
  response = client.images.generate(
    model="dall-e-3",
    prompt=str(completion.choices[0].message.content),
    size="1024x1024",
    quality="standard",
    n=1,
    response_format="b64_json",
  )
  image_url = response.data[0].url
  revisedprompt = response.data[0].revised_prompt

  prompt = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": "your job is to respond with just an appropriate title to a picture generated with the following prompt. Do not respond with anything other than the title"},
      {"role": "user", "content": revisedprompt}
    ]
  )

  title = str(prompt.choices[0].message.content)
  data_dir = Path.cwd()
  data = response.data[0].model_dump()["b64_json"]
  
  image = Image.open(BytesIO(b64decode(data)))
  W,H= image.size
  today = datetime.datetime.now()
  # Call draw Method to add 2D graphics in an image
  I1 = ImageDraw.Draw(image)
  
  # Custom font style and font size
  myFont = ImageFont.truetype('LibreBaskerville-Bold.ttf', 150)
  myFont2 = ImageFont.truetype('LibreBaskerville-Regular.ttf', 40)
  myFont3 = ImageFont.truetype('LibreBaskerville-Bold.ttf', 30)

  w = I1.textlength(title, font=myFont3)
  # Add Text to an image
  I1.text((10, 120), today.strftime("%d"), font=myFont, fill =(255, 255, 255))
  I1.text((10, 270), today.strftime("%B"), font=myFont2, fill =(255, 255, 255))
  I1.text(((W-w)/2, H-220), title, font=myFont3, fill =(255, 255, 255))

  image.save("demo.png")
  #image.save("demo.png")


if __name__ == "__main__":
  main()
