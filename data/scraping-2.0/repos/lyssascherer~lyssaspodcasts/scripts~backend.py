import modal


def download_whisper():
    # Load the Whisper model
    import os
    import whisper

    print("Download the Whisper model")

    # Perform download only once and save to Container storage
    whisper._download(whisper._MODELS["medium"], "/content/podcast/", False)


stub = modal.Stub("lyssaspodcast")
podcast_image = (
    modal.Image.debian_slim()
    .pip_install(
        "feedparser",
        "https://github.com/openai/whisper/archive/9f70a352f9f8630ab3aa0d06af5cb9532bd8c21d.tar.gz",
        "requests",
        "ffmpeg",
        "openai",
        "tiktoken",
    )
    .apt_install("ffmpeg")
    .run_function(download_whisper)
)


@stub.function(image=podcast_image, gpu="any", timeout=600)
def get_transcribe_podcast(local_path, episode_url, episode_name):
    print("Starting Podcast Transcription Function")
    print("Episode name: ", episode_name)
    print("Local Path:", local_path)
    print("RSS URL read and episode URL: ", episode_url)

    # Download the podcast episode by parsing the RSS feed
    from pathlib import Path

    p = Path(local_path)
    p.mkdir(exist_ok=True)

    print("Downloading the podcast episode")
    import requests

    with requests.get(episode_url, stream=True) as r:
        r.raise_for_status()
        episode_path = p.joinpath(episode_name)
        with open(episode_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print("Podcast Episode downloaded")

    # Load the Whisper model
    import os
    import whisper

    # Load model from saved location
    print("Load the Whisper model")
    model = whisper.load_model(
        "medium", device="cuda", download_root="/content/podcast/"
    )

    # Perform the transcription
    print("Starting podcast transcription")
    result = model.transcribe(local_path + episode_name)

    # Return the transcribed text
    print("Podcast transcription completed, returning results...")

    return result["text"]


@stub.function(image=podcast_image)
def get_token_trranscription(episode_transcription):
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    n_tokens = len(enc.encode(episode_transcription))
    print("Number of tokens in transcription ", n_tokens)
    return n_tokens


@stub.function(image=podcast_image)
def get_token_length(text):
    import tiktoken

    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    n_tokens = len(enc.encode(text))
    print(f"Number of tokens: {n_tokens}")
    return n_tokens


@stub.function(image=podcast_image)
def choose_model(transcript, extra_tokens=800):
    n_tokens = get_token_length.remote(transcript)
    total_expected_tokens = n_tokens + extra_tokens
    if total_expected_tokens < 4096:
        return "gpt-3.5-turbo"
    elif total_expected_tokens < 16384:
        return "gpt-3.5-turbo-16k"
    else:
        return None


@stub.function(image=podcast_image, secret=modal.Secret.from_name("my-openai-secret"))
def get_completion_from_messages(
    messages, functions=None, function_call=None, model="gpt-3.5-turbo", temperature=0
):
    import openai

    try:
        response = openai.ChatCompletion.create(
            model=model, messages=messages, functions=functions, temperature=temperature
        )

        return response.choices[0].message
    except Exception as e:
        return f"OPENAI_API_ERROR - {e}"


@stub.function(image=podcast_image)
def get_additional_info_from_chatgpt(transcript):
    import json

    model = choose_model.remote(transcript, extra_tokens=800)
    delimiter = "####"

    messages = [
        {
            "role": "system",
            "content": f"Extarct information about the pocast's episode. The transcription of the podcast will be delimited by the charactere '{delimiter}'",
        },
        {"role": "user", "content": f"{delimiter}{transcript}{delimiter}"},
    ]
    functions = [
        {
            "name": "extract_podcast_information",
            "description": "Given the transcription of a podcast episode, extract information about the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "summary of the podcast mentioning guests and main topics discussed. Maximum 100 words.",
                    },
                    "hosts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "list with the name of the hosts of the episode.",
                    },
                    "guests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "list with the name of the guests on the episode. Guests are people talking with the hosts. Avoid returning names mentioned by the host that are not activly participating in the discussion.",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "list keywords about the episode. Maximum 5 keywords.",
                    },
                    "key_points": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "list with higlights of what the host and guests discussed in the episode. Maximum 5 higlights.",
                    },
                    "main_output": {
                        "type": "string",
                        "description": "main thing we can learn from the epsiode, like actionable outcomes. Maximum 3 sentences.",
                    },
                },
                "required": [
                    "summary",
                    "hosts",
                    "guests",
                    "keywords",
                    "key_points",
                    "main_output",
                ],
            },
        }
    ]
    function_call = {"name": "extract_podcast_information"}
    if model:
        response_message = get_completion_from_messages.remote(
            messages,
            functions=functions,
            function_call=function_call,
            model=model,
            temperature=0,
        )
        if "OPENAI_API_ERROR" in response_message:
            print(response_message)
            return {}
        else:
            try:
                return json.loads(response_message["function_call"]["arguments"])
            except Exception as e:
                print(f"exception: {e}")
                return {}

    else:
        print("This episode is too long, we cannot process it!")

    return {}


@stub.function(
    image=podcast_image, secret=modal.Secret.from_name("my-openai-secret"), timeout=1200
)
def process_podcast(path, episodeurl, episodename):
    episode_transcription = get_transcribe_podcast.remote(path, episodeurl, episodename)
    additional_info = get_additional_info_from_chatgpt.remote(episode_transcription)
    output = {
        "episode_name": episodename,
        "episode_transcription": episode_transcription,
    }
    output = output | additional_info
    return output


@stub.local_entrypoint()
def main(path, episodeurl, episodename):
    episode_transcription = get_transcribe_podcast.remote(path, episodeurl, episodename)
    additional_info = get_additional_info_from_chatgpt.remote(episode_transcription)
    output = {
        "episode_name": episodename,
        "episode_transcription": episode_transcription,
    }
    output = output | additional_info

    print(output)
