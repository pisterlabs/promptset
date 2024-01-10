# pipedream add-package youtube-transcript-api
# pipedream add-package beautifulsoup4
# pipedream add-package html2text

from typing import List, Dict, Tuple
from youtube_transcript_api import YouTubeTranscriptApi
import re
import requests
import openai
import html2text

MAX_LENGTH = int(4000 * 2.5)


def handler(pd: "pipedream"):
    event = pd.steps["trigger"]["event"]["message"]
    telegram_user_id = str(event["from"]["id"])
    msg = event["text"]
    if msg[:6] == "\\token":
        save_token(pd, telegram_user_id, msg[7:].strip())
        return {
            "status": 400,
            "message": "OpenAI API token registered!"
        }
    global openai_api_key
    try:
        openai_api_key = get_token(pd, telegram_user_id)
        if openai_api_key == "":
            raise ValueError
    except:
        return {
            "status": 400,
            "message": "Please register your OpenAI API key with \\token [OPENAI_TOKEN]."
        }
    subject, _url, html = create_content(msg)
    html = html.replace("\n", "<br>")
    html = f"<html><body>{html}</body></html>"

    return {
        "subject": subject,
        "text": subject,
        "url": _url,
        "html": html,
        "markdown": html2text.HTML2Text().handle(html),
        "status": 200,
        "message": "Saved on your XBrain!"
    }


def save_token(pd, user_id: str, token: str):
    print(f"Saving token for user {user_id}, token: {token}")
    pd.inputs["data_store"][user_id] = token


def get_token(pd, user_id: str):
    return pd.inputs["data_store"][user_id]


def create_content(user_msg: str) -> Tuple[str, str, str]:
    # check if the user_msg starts with a command '/'
    if user_msg[0] == "/":
        command, *content = user_msg.split(" ")
        content = " ".join(content)
    else:
        command = "/add"
        content = user_msg
    if not content:
        raise ValueError("Content is empty")

    # content is the url unless it is a text
    global MSG_URL
    MSG_URL = content
    if command in ("/sum", "/tldr", "/ask"):
        subject, html = gpt(command, content)
    else:
        if command != "/add":
            content = f"{command} {content}"
        subject, html = add(content)
    return subject, MSG_URL, html


def add(content: str) -> Tuple[str, str]:
    type = get_content_type(content)
    if type == "youtube":
        subject, html = add_youtube_url(content)
        subject = f"XBrian: {subject}"
    elif type == "webpage":
        # Emailing url to omnivore will automatically add the content
        subject = content  # subject is the url
        html = ""
    else:
        subject, *_, html = content.split("\n")
        html = "\n".join(html)
    return subject, html


def gpt(command: str, content: str) -> Tuple[str, str]:
    type = get_content_type(content)
    subject = content.split("\n")[0]

    if type == "youtube":
        url = re.findall(r"(https?://\S+)", content)[0]
        prompt = content.replace(url, "")
        subject = f"XBrian: {subject}"
        content = YoutubeTranscriptApiHelper().get_transcript(url)
        content = f"{prompt} {content}"
    elif type == "webpage":
        url = re.findall(r"(https?://\S+)", content)[0]
        prompt = content.replace(url, "")
        subject, content = HtmlHelper().download_html(url)
        subject = f"XBrian: {subject}"
        content = f"{prompt} {content}"

    return subject, get_assistant(content, command)


def add_webpage_url(url: str) -> Tuple[str, str]:
    htmlHelper = HtmlHelper()
    subject, html = htmlHelper.download_html(url)
    return subject, f"{html}{htmlHelper.add_html_hyperlist(url)}"


def add_youtube_url(url: str) -> Tuple[str, str]:
    htmlHelper = HtmlHelper()
    subject = htmlHelper.get_pagetitle(url)
    transcript = YoutubeTranscriptApiHelper().get_transcript(url)
    return subject, f"{htmlHelper.add_html_hyperlist(url)}{transcript}"


def get_content_type(content: str) -> str:
    if "youtu.be" in content or "youtube.com" in content:
        return "youtube"
    elif re.match(r"^https?://", content):
        return "webpage"
    else:
        global MSG_URL
        MSG_URL = "No URL found"
        return "text"


def get_assistant(text: str, command: str, openapi_token: str = "") -> str:
    if len(text) > MAX_LENGTH:
        print(f"WARNING: Content is too long. Length: {len(text)}")
        text = text[:MAX_LENGTH]

    if openapi_token == "":
        global openai_api_key
        openapi_token = openai_api_key
    openai_helper = OpenAIHelper(openapi_token)
    return openai_helper.chat(text, command)

# =========================================================================================================
# ====================================            HtmlHelper           ====================================
# =========================================================================================================


class HtmlHelper:
    def get_pagetitle(self, url, html=""):
        if html == "" or "<title>" not in html:
            return url
        else:
            return html.split("<title>")[1].split("</title>")[0]

    def add_html_hyperlist(self, link):
        return f"<br><br><a href='{link}'>URL: {link}</a><br><br>"

    def download_html(self, url):
        try:
            html = requests.get(url).text
            subject = self.get_pagetitle(url, html)
            body = html.split("<body>")[1].split("</body>")[0]
            # remove all the svg, img, script, style, link, meta, noscript
            body = re.sub(r"<svg.*?</svg>", "", body, flags=re.DOTALL)
            body = re.sub(r"<img.*?>", "", body, flags=re.DOTALL)
            body = re.sub(r"<script.*?</script>", "", body, flags=re.DOTALL)
            body = re.sub(r"<style.*?</style>", "", body, flags=re.DOTALL)
            body = re.sub(r"<link.*?>", "", body, flags=re.DOTALL)
            body = re.sub(r"<meta.*?>", "", body, flags=re.DOTALL)
            body = re.sub(r"<noscript.*?</noscript>",
                          "", body, flags=re.DOTALL)
            return subject, body
        except Exception as e:
            print(f"ERROR: {e}")
            return url, "ERROR: Unable to download the webpage."

# =========================================================================================================
# ====================================            OpenAIHelper           ==================================
# =========================================================================================================


class OpenAIHelper:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key

    def call_chatgpt(
        self,
        messages: List[Dict[str, str]],
        temperature: int = 0,
        model: str = "gpt-3.5-turbo",
    ) -> str:
        # temperature: What sampling temperature to use, between 0 and 2.
        # Higher values like 0.8 will make the output more random,
        # while lower values like 0.2 will make it more focused and deterministic.
        return openai.ChatCompletion.create(
            model=model, temperature=temperature, max_tokens=100, messages=messages
        )

    def get_system_prompt(self):
        return [
            {
                "role": "system",
                "content": (
                    "As a text analysis assistant, analyzing content from various articles."
                ),
            },
        ]

    def get_user_prompt_tldr(self, content: str):
        return {
            "role": "user",
            "content": (
                f"In the same language, write TLDR, in {MAX_LENGTH} characters or less, capturing all important details:\n```{content}```"
            ),
        }

    def get_user_prompt_ask(self, content: str):
        return {"role": "user", "content": (f"Answer my question: {content}")}

    def chat(self, text, command):
        prompt = self.get_system_prompt()
        if command == "/sum" or command == "/tldr":
            prompt.append(self.get_user_prompt_tldr(text))
        else:
            prompt.append(self.get_user_prompt_ask(text))
        response = self.call_chatgpt(prompt)
        return response.choices[0].message.content

# =========================================================================================================
# ==============================            YoutubeTranscriptApiHelper           ==========================
# =========================================================================================================


class YoutubeTranscriptApiHelper:
    def get_video_id_from_url(self, url):
        video_id = None
        pattern = re.compile(
            r"(?:http(?:s)?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9-_]+)"
        )
        match = pattern.search(url)
        if match:
            video_id = match.group(1)
        return video_id

    def get_transcript_from_id(self, video_id):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return transcript
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_transcript(self, url):
        video_id = self.get_video_id_from_url(url)
        if video_id:
            transcript = self.get_transcript_from_id(video_id)
            if transcript:
                return " ".join([entry["text"] for entry in transcript])
            else:
                raise Exception("No transcript found for the video.")
        else:
            raise Exception("Invalid YouTube URL.")
