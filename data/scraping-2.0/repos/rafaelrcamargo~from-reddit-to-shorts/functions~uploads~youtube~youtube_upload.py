import os
from datetime import datetime
from pathlib import Path
from googleapiclient.http import MediaFileUpload

from functions.uploads.youtube.google import Create_Service
from functions.utils.separator import separator

# Cool Terminal Colors
from rich import print
from rich.console import Console


# .env file
from dotenv import load_dotenv

# OpenAI API
import openai

from functions.utils.timeout import timeout

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

CLIENT_SECRET_FILE = (
    str(Path(__file__).cwd()) + "/functions/uploads/youtube/client_secret.json"
)
API_NAME = "youtube"
API_VERSION = "v3"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

console = Console()

build_path = (
    str(Path(__file__).cwd()) + "/temp/build/" + datetime.today().strftime("%d_%m_%Y")
)

ATTEMPTS = 10


def youtube_upload(filename):
    print("\n>> [yellow]Success logging in[/yellow], starting the [blue]upload![/blue]")

    global ATTEMPTS

    if filename.split(".")[1] == "mp4":
        print(f"\n>> [yellow]File[/yellow]: {filename}")

        # get last item from array
        name = filename.split("/")[-1].split(".")[0]
        subreddit = "r/" + filename.split("/")[-1].split("_")[0]

        completion = openai.Completion.create(
            model="text-curie-001",
            prompt=f"Write a viral funny clickbait title for a Youtube Shorts about the trending subreddit {subreddit} memes compilation:\n",
            temperature=0.8,
            max_tokens=128,
            top_p=1,
            best_of=4,
            frequency_penalty=0.5,
            presence_penalty=0.25,
        )

        print(
            f">> [yellow]Prompt[/yellow]:",
            f"Write a viral funny clickbait title for a Youtube Shorts about the trending subreddit {subreddit} memes compilation.",
        )

        print("\n")
        console.print_json(str(completion))

        title = completion.choices[0].text
        title = title.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()

        print(f"\n>> [yellow]Name[/yellow]: {name}")
        print(f">> [yellow]Subreddit[/yellow]: {subreddit}")
        print(f">> [yellow]Title[/yellow]: {title}")

        try:
            request_body = {
                "snippet": {
                    "categoryId": 24,
                    "title": title + f" | {subreddit} #Shorts",
                    "description": title
                    + " | "
                    + subreddit
                    + " 🤣 | #Shorts\n\n"
                    + "Subscribe for a cookie 🍪"
                    + "\n\n#fun, #funny, #comedy, #meme, #trending, #memes, #nonsense, #reddit, #viral",
                    "tags": [
                        "fun",
                        "funny",
                        "comedy",
                        "trending",
                        "meme",
                        "memes",
                        "nonsense",
                        "reddit",
                        "youtube",
                        "viral",
                        "Shorts",
                    ],
                },
                "status": {
                    "privacyStatus": "public",
                    "selfDeclaredMadeForKids": False,
                },
                "notifySubscribers": True,
            }

            mediaFile = MediaFileUpload(build_path + "/" + name + ".mp4")

            try:
                response_upload = (
                    service.videos()
                    .insert(
                        part="snippet,status", body=request_body, media_body=mediaFile
                    )
                    .execute()
                )
                print("\n")
                console.print_json(str(response_upload))
                print("\n>> [blue]Uploaded![/blue]")
            except Exception as e:
                if ATTEMPTS <= 0:
                    return

                print(f"\n>> [red]Upload failed![/red] {str(e)}")
                print(f"\n>> Trying again. ({str(ATTEMPTS)} attempts left)")

                timeout(30, 1, "request")
                print("\n", separator(), "\n")

                ATTEMPTS -= 1
                youtube_upload(filename)
        except Exception as e:
            print(f"\n>> [red]Error: {str(e)}[/red]")
            print("\n", separator(), "\n")
    else:
        print("\n", separator(), "\n")
        print(
            f">> [red]Why [bold]TF[/bold] is this file here?[/red]\n>> [red]File[/red]:{filename}"
        )
        print("\n", separator(), "\n")
