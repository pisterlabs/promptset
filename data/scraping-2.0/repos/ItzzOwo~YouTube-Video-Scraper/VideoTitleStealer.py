#to get a channel id, use this site: https://commentpicker.com/youtube-channel-id.php
#you will have to replace the API key with your own at https://developers.google.com/youtube/v3
#if you encounter any problems, join my discord: https://discord.gg/WW5PuDBySt

import requests
import html
import openai

def get_video_titles(api_key, channel_id, max_results=500): #change max results here if you want more or less videos
    try:
        base_url = "https://www.googleapis.com/youtube/v3/search"
        titles = []
        next_page_token = None

        while len(titles) < max_results:
            params = {
                "key": api_key,
                "part": "snippet",
                "channelId": channel_id,
                "maxResults": min(50, max_results - len(titles)),
                "pageToken": next_page_token,
            }

            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print("Failed to fetch video data from YouTube API.")
                break

            data = response.json()
            titles += [item["snippet"]["title"] for item in data["items"]]
            next_page_token = data.get("nextPageToken")

            if not next_page_token:
                break

        return titles
    except Exception as e:
        print("An error occurred:", e)
        return []

def rewrite_titles(titles):
    try:
        openai.api_key = ""  #replace api key with openai api key
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt="\n".join(titles),
            temperature=0.7,
            max_tokens=200,
            n=1,
        )

        rewritten_titles = response['choices'][0]['text'].strip().split("\n")
        return rewritten_titles
    except Exception as e:
        print("An error occurred while rewriting titles:", e)
        return []

def scrape_youtube_titles(channel_id):
    try:
        api_key = "youtube"  #replace with youtube dev api key at https://developers.google.com/youtube/v3
        openai.api_key = "openai"  #replace with openai api key

        titles = get_video_titles(api_key, channel_id)
        titles = [html.unescape(title) for title in titles]

        create_unique_sentences = input("Would you like to rewrite the titles? (yes/no): ").lower().strip()
        if create_unique_sentences == "yes":
            rewritten_titles = rewrite_titles(titles)
            if rewritten_titles:
                save_options = input("Do you want to save the original titles, rewritten titles, or both? (original/rewritten/both): ").lower().strip()
                if save_options in ['original', 'both']:
                    with open("video_titles.txt", "w", encoding="utf-8") as file:
                        for title in titles:
                            file.write(title + "\n")
                    print(f"Successfully scraped and saved {len(titles)} original video titles to 'video_titles.txt'")

                if save_options in ['rewritten', 'both']:
                    with open("rewritten_video_titles.txt", "w", encoding="utf-8") as file:
                        for title in rewritten_titles:
                            file.write(title + "\n")
                    print(f"Successfully rewrote {len(rewritten_titles)} video titles and saved them to 'rewritten_video_titles.txt'")
            else:
                print("Failed to rewrite titles using the ChatGPT API.")
        else:
            with open("video_titles.txt", "w", encoding="utf-8") as file:
                for title in titles:
                    file.write(title + "\n")
            print(f"Successfully scraped {len(titles)} video titles and saved them to 'video_titles.txt'")
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    youtube_channel_id = input("Enter the YouTube channel ID (e.g., UC1234567890): ")
    scrape_youtube_titles(youtube_channel_id)
