import modal

def download_whisperX():
  # Load the WhisperX model
  import whisperx
  print ("Download the WhisperX model")

  # Parameters for whisperX
  device = "cpu"
  compute_type = "float32"

  # Perform download only once and save to Container storage
  _ = whisperx.load_model("medium", device, compute_type=compute_type)

stub = modal.Stub("podcast-project")
podcast_image = modal.Image.debian_slim().pip_install("feedparser",
                                                     "requests",
                                                     "ffmpeg",
                                                     "openai",
                                                     "tiktoken",
                                                     "wikipedia",
                                                     "ffmpeg-python",
                                                     "googlesearch-python"
                                                     ).apt_install("git","ffmpeg")

podcast_image = podcast_image.pip_install("torch",
                                        "torchvision",
                                        "torchaudio",
                                        index_url="https://download.pytorch.org/whl/cu118")

podcast_image = podcast_image.pip_install("git+https://github.com/m-bain/whisperx.git"
                                        ).run_function(download_whisperX)


@stub.function(image=podcast_image, gpu="any", timeout=600)
def get_transcribe_podcast(rss_url, local_path):
  print ("Starting Podcast Transcription Function")
  print ("Feed URL: ", rss_url)
  print ("Local Path:", local_path)

  # Read from the RSS Feed URL
  import feedparser
  intelligence_feed = feedparser.parse(rss_url)
  podcast_title = intelligence_feed['feed']['title']
  episode_title = intelligence_feed.entries[0]['title']
  episode_image = intelligence_feed['feed']['image'].href
  for item in intelligence_feed.entries[0].links:
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

  print ("Podcast Episode downloaded")

  # Load the Whisper model
  import os
  import whisperx

  # Parameters for whisperX
  device = "cuda"
  batch_size = 32 # reduce if low on GPU mem
  compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

  # Load model from saved location
  print ("Load the Whisper model")
  model = whisperx.load_model("medium", device=device, compute_type=compute_type)

  # Get the path for audio file
  audio = whisperx.load_audio(local_path + episode_name)

  # Perform the transcription
  print ("Starting podcast transcription")
  result = model.transcribe(audio, batch_size=batch_size)

  # Combine result text
  result["text"] = ""
  for segments in result["segments"]:
    result["text"] = result["text"] + segments["text"]

  # Return the transcribed text
  print ("Podcast transcription completed, returning results...")
  output = {}
  output['podcast_title'] = podcast_title
  output['episode_title'] = episode_title
  output['episode_image'] = episode_image
  output['episode_transcript'] = result['text']
  return output

@stub.function(image=podcast_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_summary(podcast_transcript):
  import openai
  # Check token size
  import tiktoken
  enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k-0613")
  num_tokens = len(enc.encode(podcast_transcript))
  if num_tokens >= 16385:
    return ""
  
  instructPrompt = """
  You are an expert copywriter who is responsible for publishing newsletters with thousands of subscribers. You recently listened to a great podcast
  and want to share a summary of it with your readers. Please write the summary of this podcast making sure to cover the important aspects that were
  discussed. Do not include introductions and sponsorship details.Readers want to know the outlines of the topic being covered.
  You always need to get the podcast summary. Always continue extracting for the transcript of the podcast I provide below.
  Please keep it concise and meaningful.
  The transcript of the podcast is provided below.\n
  """
  request = instructPrompt + podcast_transcript

  chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k-0613",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ],
                                            max_tokens=256
                                            )
  podcastSummary = chatOutput.choices[0].message.content
  return podcastSummary

@stub.function(image=podcast_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_guest(podcast_transcript):
  import openai
  import wikipedia
  import json
  from googlesearch import search

  request = podcast_transcript[:15000] # first 15k characters
  
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-16k-0613",
    messages=[{"role": "user", "content": request}],
    max_tokens=256,
    functions=[
    {
        "name": "get_podcast_guest_information",
        "description": "Get information on the podcast guest using their name and job to search on Google",
        "parameters": {
            "type": "object",
            "properties": {
                "guest_name": {
                    "type": "string",
                    "description": "The name of the guest who is speaking in the podcast",
                },
                "unit": {"type": "string"},
                "guest_job": {
                    "type": "string",
                    "description": "The job of the guest who is speaking in the podcast",
                },
                "unit": {"type": "string"},
            },
            "required": ["guest_name", "guest_job"],
        },
    }
    ],
    function_call={"name": "get_podcast_guest_information"}
    )
  podcast_guest = ""
  podcast_guest_job = ""
  response_message = completion["choices"][0]["message"]
  if response_message.get("function_call"):
      function_name = response_message["function_call"]["name"]
      function_args = json.loads(response_message["function_call"]["arguments"])
      podcast_guest=function_args.get("guest_name")
      podcast_guest_job=function_args.get("guest_job")

  if podcast_guest_job is None:
    podcast_guest_job = ""

  query = '%s %s' % (podcast_guest, podcast_guest_job)
  
  wiki_title, wiki_summary, wiki_url, wiki_img = get_wiki_info(query)
  
  search_results = []
  try:
    for response in search(query):
        search_results.append(response)
  except:
    search_results[0] = ""

  podcastGuest = {}
  podcastGuest['name'] = podcast_guest
  podcastGuest['job'] = podcast_guest_job
  podcastGuest['wiki_title'] = wiki_title
  podcastGuest['wiki_summary'] = wiki_summary
  podcastGuest['wiki_url'] = wiki_url
  podcastGuest['wiki_img'] = wiki_img
  podcastGuest['google_URL'] = search_results[0]

  return podcastGuest

@stub.function(image=podcast_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_podcast_highlights(podcast_transcript):
  import openai
  # Check token size
  import tiktoken
  enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k-0613")
  num_tokens = len(enc.encode(podcast_transcript))
  if num_tokens >= 16385:
    return ""
  instructPrompt = """
  You are a podcast editor and producer. You help audience to understand the podcast deeply.
  You are provided with the transcript of a podcast episode and have to identify the 5 most significant moments in the podcast as highlights.
  - Each highlight needs to be a statement by one of the podcast guests
  - Each highlight has to be impactful and an important takeaway from this podcast episode
  - Each highlight must be concise and make listeners want to hear more about why the podcast guest said that
  - The highlights that you pick must be spread out throughout the episode
  
  Provide only the highlights and nothing else. Provide the full sentence of the highlight and format it as follows:\n
  - Highlight 1 of the podcast
  - Highlight 2 of the podcast
  - Highlight 3 of the podcast
  - Highlight 4 of the podcast
  - Highlight 5 of the podcast
  
  The transcript of the podcast episode is below. Use the transcript in order to extract highlights.
  Provide only the highlights and nothing else. Do not include your sentence, only the highlights.\n
  """

  request = instructPrompt + podcast_transcript
  chatOutput = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k-0613",
                                            messages=[{"role": "system", "content": "You are a helpful assistant."},
                                                      {"role": "user", "content": request}
                                                      ],
                                            max_tokens=256,
                                            )
  podcastHighlights = chatOutput.choices[0].message.content
  return podcastHighlights

@stub.function(image=podcast_image, secret=modal.Secret.from_name("my-openai-secret"), timeout=1200)
def process_podcast(url, path):
  output = {}
  podcast_details = get_transcribe_podcast.remote(url, path)
  podcast_summary = get_podcast_summary.remote(podcast_details['episode_transcript'])
  podcast_guest = get_podcast_guest.remote(podcast_details['episode_transcript'])
  podcast_highlights = get_podcast_highlights.remote(podcast_details['episode_transcript'])
  output['podcast_details'] = podcast_details
  output['podcast_summary'] = podcast_summary
  output['podcast_guest'] = podcast_guest
  output['podcast_highlights'] = podcast_highlights
  return output

def get_wiki_info(search_term):
  import wikipedia
  import requests
  import json

  try:
    print ("Searching Wikipedia for guest...")
    result = wikipedia.search(search_term, results = 1)
    wikipedia.set_lang('en')
    wkpage = wikipedia.WikipediaPage(title = result[0])
    title = wkpage.title
    url = wkpage.url
    summary = wkpage.summary
    response  = requests.get('http://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles='+title)
    json_data = json.loads(response.text)
    try:
      img_link = list(json_data['query']['pages'].values())[0]['original']['source']
    except:
      img_link = ""
    return (title, summary, url, img_link)
  except wikipedia.exceptions.PageError:
    print('The page for guest does not exist on Wikipedia.')
    return ("", "", "", "")
  except wikipedia.exceptions.DisambiguationError as e:
    print(f'The page for guest is ambiguous. Possible matches are:')
    print(e.options)
    return ("", "", "", "")
  except:
      return ("", "", "", "")

@stub.local_entrypoint()
def test_method(url, path):
  output = {}
  podcast_details = get_transcribe_podcast.remote(url, path)
  print ("Podcast Summary: ", get_podcast_summary.remote(podcast_details['episode_transcript']))
  print ("Podcast Guest Information: ", get_podcast_guest.remote(podcast_details['episode_transcript']))
  print ("Podcast Highlights: ", get_podcast_highlights.remote(podcast_details['episode_transcript']))