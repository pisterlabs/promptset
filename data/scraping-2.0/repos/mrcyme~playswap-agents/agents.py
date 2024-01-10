import datetime
import json
from utils.playswap_api_wrapper import PlayswapWrapper
from utils.utils import *
from scrappers import *
from openai import OpenAI
import yaml



with open('config.yaml') as f:
    config = yaml.safe_load(f)

OPENAI_API_KEY = config['OPENAI_API_KEY']

with open("./utils/tools.json", 'r') as j:
    TOOLS = json.loads(j.read())

client = OpenAI(api_key=OPENAI_API_KEY)
    

class PlaylisterAgent():
    def __init__(self, scrapper, model_name, playswap_token, language="en", publish=False) -> None:
        self.scrapper = scrapper
        self.model_name = model_name
        self.language = language
        self.text = self.scrap()
        self.playlist_candidates = []
        self.publish = publish
        self.playswap = PlayswapWrapper(playswap_token)


    def scrap(self):
        return self.scrapper.scrap()

    def get_playlist_ideas(self):
        pass

    def send_to_playswap(self):
        for p in self.playlist_candidates:
            print(p)
            id = self.playswap.create_playlist(p["title"], p["description"])["id"]
            for influence in p["influences"]:
                resp = self.playswap.create_influence(id, url_to_id(influence, 'playlist'))
                print(resp)

    def add_influences(self):
        for p in self.playlist_candidates:
            if self.target=="album":
                p["influences"] = google_search_playlists(f"{p['artist']}")

            elif self.target=="movie":            
                p["influences"] = google_search_playlists(f"{p['movie']}")

    def run(self):
        pass



class AlbumPlaylisterAgent(PlaylisterAgent):
    """
    Agent for creating playlist from unreleased albums
    """
    def __init__(self, scrapper, model_name, playswap_token, language="en", publish=False) -> None:
        super().__init__(scrapper, model_name, playswap_token, language=language, publish=publish)
        self.target = "album"

    def get_album_list(self):
        message = f"""
        Can you get a list of all the unreleased album from the following text, for each album give the the title, the artist and the relase date, sort album by descending date. Unreleased means that the release date is after {datetime.datetime.now().year}, please respond in json format. {self.text}
        """
        messages = [{"role": "user", "content": f"{message}"}]
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={ "type": "json_object" },
            tools=TOOLS["get_album_list_from_text"],
            tool_choice="auto",  # auto is default, but we'll be explicit
        )
        album_list = json.loads(response.choices[0].message.tool_calls[0].function.arguments)["albums"]
        #Filter response that does not contain all the info needed
        album_list = remove_items_missing_keys(album_list, ["title", "artist", "date"])
        return album_list


    def run(self):
        album_list = self.get_album_list()
        print(album_list)
        unreleased_album_list = parse_release_list(album_list)
        print(unreleased_album_list)
        for a in unreleased_album_list:
            title, description = create_playlist_title_and_description("en", f"Create a playlist for {a['artist']}, {a['title']}")
            image = google_search_image(f"{a['artist']}, {a['title']}")
            self.playlist_candidates.append({"title": title, "description": description, "image": image, "artist": a["artist"], "album": a["title"]})
        self.add_influences()
        print(self.playlist_candidates)
        if self.publish: 
            self.send_to_playswap()
    
    
    
class MoviePlaylisterAgent(PlaylisterAgent):
    """
    Agent for creating playlist from unreleased movie
    """
    def __init__(self, scrapper, model_name, playswap_token, language="en", publish=False) -> None:
        super().__init__(scrapper, model_name, playswap_token, language=language, publish=publish)
        self.target = "movie"

    def get_movie_list(self, max_results=5):
        message = f"""
        Can you get a list of all the unreleased movie from the following text, for each movie give the the title, the director and the relase date, sort album by descending date. Unreleased means that the release date is after {datetime.datetime.now().year}, please respond in json format. {self.text}
        """
        messages = [{"role": "user", "content": f"{message}"}]
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={ "type": "json_object" },
            tools=TOOLS["get_movie_list_from_text"],
            tool_choice="auto",  # auto is default, but we'll be explicit
        )
        movie_list = json.loads(response.choices[0].message.tool_calls[0].function.arguments)["movies"][0:max_results]
        #Filter response that does not contain all the info needed
        movie_list = remove_items_missing_keys(movie_list, ["title", "actors", "date"])
        print(movie_list)
        return movie_list

    
    def run(self):
        movie_list = self.get_movie_list()
        unreleased_movie_list = parse_release_list(movie_list)
        for m in unreleased_movie_list:
            title, description = create_playlist_title_and_description("en", f"Create a playlist description for the movie {m['title']} with actors {','.join(m['actors'])}")
            image = google_search_image(f"{m['title']}")
            self.playlist_candidates.append({"title": title, "description": description, "image": image, "movie": m["title"], "actors": m["actors"]})
        self.add_influences()
        print(self.playlist_candidates)
        if self.publish: 
            self.send_to_playswap()


class ChartPlaylisterAgent(PlaylisterAgent):
    pass

def create_playlist_title_and_description(language, prompt, model="gpt-3.5-turbo-1106"):
    messages = []
    system_prompt = f"""
    You are a super smart music marketting expert that create engaging playlist title and description. The user can ask you to create playlist title in french or in english. The user can also give specific instruction on what to include in the description. please return the response in json with tilte and description as keys.
    """
    messages.append({"role": "system", "content": system_prompt})
    if language == "fr":
        messages.append({"role": "user", "content": f"{prompt} , please create title and description in french"})
    elif language=="en":
        messages.append({"role": "user", "content": f"{prompt} , please create title and description in english"})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={ "type": "json_object" },
    )
    title = json.loads(response.choices[0].message.content)["title"]
    description = json.loads(response.choices[0].message.content)["description"][0:100]
    return title, description


"""
scrapper = Bs4Scrapper("https://www.trackmusik.fr/media/albums-rap-francais")
agent = AlbumPlaylisterAgent(scrapper, "gpt-3.5-turbo-1106", PLAYSWAP_API_KEY, language="fr", publish=False)
agent.run()


scrapper = Bs4Scrapper("https://www.imdb.com/calendar/")
agent = MoviePlaylisterAgent(scrapper, "gpt-3.5-turbo-1106", PLAYSWAP_API_KEY, language="en", publish=False)
agent.run()


playswap = PlayswapWrapper(PLAYSWAP_API_KEY)
id = playswap.create_playlist("zouloubinks", "test")["id"]

influences = ['https://open.spotify.com/playlist/37i9dQZF1DZ06evO4tMsQE', 'https://open.spotify.com/playlist/37i9dQZF1EIUYeydRFbHYG', 'https://open.spotify.com/playlist/37i9dQZF1DZ06evO2cNShi']
for influence in influences:
    resp = playswap.create_influence(id, url_to_id(influence, 'playlist'))
    print(resp)

"""
