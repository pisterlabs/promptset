import modal

def download_whisper():
  # Load the Whisper model
  import os
  import whisper
  print ("Download the Whisper model")

  # Perform download only once and save to Container storage
  whisper._download(whisper._MODELS["medium"], '/content/podcast/', False)


stub = modal.Stub("corise-podcast-project")
corise_image = modal.Image.debian_slim().pip_install("feedparser",
                                                     "https://github.com/openai/whisper/archive/9f70a352f9f8630ab3aa0d06af5cb9532bd8c21d.tar.gz",
                                                     "requests",
                                                     "ffmpeg",
                                                     "openai",
                                                     "tiktoken",
                                                     "ffmpeg-python").apt_install("ffmpeg").run_function(download_whisper)

@stub.function(image=corise_image, gpu="any", timeout=600)
def get_transcribe_podcast(rss_url, local_path,num):
  print ("Starting Podcast Transcription Function")
  print ("Feed URL: ", rss_url)
  print ("Local Path:", local_path)

  # Read from the RSS Feed URL
  import feedparser
  mp_feed = feedparser.parse(rss_url)
  podcast_title = mp_feed['feed']['title']
  episode_title = mp_feed.entries[1]['title']
  episode_image = mp_feed['feed']['image'].href
  episode_num = int(num)
  for item in mp_feed.entries[episode_num].links:
    if (item['type'] == 'audio/mpeg'):
      episode_url = item.href
  episode_name = "podcast_episode.mp3"
  print ("RSS URL read and episode URL: ", episode_url)

  # Download the podcast episode by parsing the RSS feed
  from pathlib import Path
  p = Path(local_path)
  p.mkdir(exist_ok=True)

  print ("Downloading the podcast episode")
  import requests
  with requests.get(episode_url, stream=True) as r:
    r.raise_for_status()
    episode_path = p.joinpath(episode_name)
    with open(episode_path, 'wb') as f:
      for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

  print("Podcast Episode downloaded")

  # Load the Whisper model
  import os
  import whisper

  # Load model from saved location
  print ("Load the Whisper model")
  model = whisper.load_model('medium', device='cuda', download_root='/content/podcast/')

  # Perform the transcription
  print ("Starting podcast transcription")
  result = model.transcribe(local_path + episode_name)

  # Return the transcribed text
  print ("Podcast transcription completed, returning results...")
  output = {}
  output['podcast_title'] = podcast_title
  output['episode_title'] = episode_title
  output['episode_image'] = episode_image
  output['episode_transcript'] = result['text']
  return output

@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_summary(podcast_transcript):
  import openai

  instructPrompt = """
  You are an expert copywriter and economist who is responsible for publishing newsletters about economics' podcasts with thousands of subscribers.
  
  Create a summary of the following podcast. 
  
  Please adhere to the following guidelines in your summary: 
  
  1) Your summary should be comprehensive but concise and should include all the topics touched upon in the podcast.
  2) For Marketplace podcasts, ignore all the advertisements for companies or information about other Marketplace podcast series in the summary. For other podcasts, ignore all the advertisements. 
  3) Keep your summary within 200-250 words. 
  4) Make sure the summary is easy to follow and understandable to a reader with limited economics background.
  5) Keep the summary high-level and topical, do not mention the names of the podcast guests.
  """
  
  request = instructPrompt + podcast_transcript
  
  chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ]
                                            )
  
  podcastSummary = chatOutput.choices[0].message.content
  return podcastSummary

@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_guest(podcast_transcript):
  import openai
  import json

  function_descriptions= [
    {
        "name": "get_podcast_guest_information",
        "description": "This function searches for the names and occupations of podcast guests on Google. It accepts an array of objects. Each object should include another guests' name. Make sure to grab all the names mentioned in the podcast.",
        "parameters": {
            "type": "object",
            "properties": {
                "guest_names":{
                    "type" : "array",
                    "items": {
                        "type":"object",
                        "properties": {
                            "guestname" : {
                                "type": "string",
                                "description": "The name and surname of the guest, e.g. olu sinola",
                    },
                            "occupation": {
                                "type": "string",
                                "description": "The occupation of the guest, e.g. global head of strategy",     
                            },
                            "company": {
                                "type": "string",
                                "description": "The company of the guest, e.g. credit sites",
                            }
                },
                        "required": ["guestname","occupation","company"],
                    },        
            }
                },
            "required": ["guest_names"],
        },
}
    ]

  request = podcast_transcript[200:-300]
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k",
    messages=[{"role": "user", "content": request}],
    functions=function_descriptions,
    function_call={"name": "get_podcast_guest_information"},
    temperature=0)
    
  response_message = completion["choices"][0]["message"]
  if response_message.get("function_call"):
    function_name = response_message["function_call"]["name"]
    function_args = json.loads(response_message["function_call"]["arguments"])
    guest_list = function_args["guest_names"]

  guest_list = [d for d in guest_list if (d.get("company") != "Marketplace" and d.get("company") != "Planet Money")]

  separator = "\n"
  
  default_company = "N/A"  # Default value for 'company' key
  
  podcastGuest = separator.join([
    f"Name: {d['guestname']}, Occupation: {d.get('occupation', default_company)}, Company: {d.get('company', default_company)}"
    for d in guest_list
    ])
    
  return podcastGuest


@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_highlights(podcast_transcript):
  import openai
  instructPrompt = """
  You are an expert copywriter and economist who is responsible for publishing newsletters about economics' podcasts with thousands of subscribers.
  
  Extract the top five key moments in the podcast. Key moments in a podcast refer to specific segments or points in the podcast episode that are particularly noteworthy, interesting, or relevant. These moments can capture important discussions, insights, jokes, or any content that stands out and could be of interest to listeners. 
  
  Your response should be formatted in bullet points. Make sure to report only five bullet points. 
  """
  
  request = instructPrompt + podcast_transcript

  chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ]
                                            )

  podcastHighlights = chatOutput.choices[0].message.content
  return podcastHighlights


@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_market_information(podcast_transcript,podcast_title):
  import openai
  if podcast_title == "Marketplace":
  
    instructPrompt = """
    You are a market specialist who is responsible for reporting the stock market price changes to a thousands of subscribers.
  
    Focus exclusively on the part of the podcast where they discuss the performance of the American stock exchanges such as the Dow, NASDAQ, and S&P 500. 
  
    Report the performance of the American stock exchanges, including the change in points, percentage change and closing index level (do not include $ sign in front of it). The output should look like this:
  
    Dow Industrial:
  
    NASDAQ:
  
    S&P 500:
    Do not include any other information (like the summary of the podcast) from the podcast.
    """
  
    request = instructPrompt + podcast_transcript

    chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ],
                                                      temperature=0
                                            )


    IndexLevels = chatOutput.choices[0].message.content

  else:
    IndexLevels = "The podcast does not contain information on the stock market performance."
  
  return IndexLevels


@stub.function(image=corise_image, secret=modal.Secret.from_name("my-openai-secret"), timeout=1200)
def process_podcast(url, path,num):
  output = {}
  podcast_details = get_transcribe_podcast.call(url, path,num)
  podcast_summary = get_podcast_summary.call(podcast_details['episode_transcript'])
  podcast_guest = get_podcast_guest.call(podcast_details['episode_transcript'])
  podcast_highlights = get_podcast_highlights.call(podcast_details['episode_transcript'])
  podcast_market = get_market_information.call(podcast_details['episode_transcript'],podcast_details['podcast_title'])
  output['podcast_details'] = podcast_details
  output['podcast_guest'] = podcast_guest
  output['podcast_summary'] = podcast_summary
  output['podcast_highlights'] = podcast_highlights
  output['podcast_market'] = podcast_market
  return output

@stub.local_entrypoint()
def test_method(url, path,num):
  output = {}
  podcast_details = get_transcribe_podcast.call(url, path,num)
  print ("Podcast Summary: ", get_podcast_summary.call(podcast_details['episode_transcript']))
  print ("Podcast Guest Information: ", get_podcast_guest.call(podcast_details['episode_transcript']))
  print ("Podcast Highlights: ", get_podcast_highlights.call(podcast_details['episode_transcript']))
  print ("Podcast Market Information: ", get_market_information.call(podcast_details['episode_transcript'],podcast_details['podcast_title']))
