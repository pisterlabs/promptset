#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai


class Gepeto:
    def __init__(self):
        openai.api_type = "azure"
        openai.api_base = "https://gepeto.openai.azure.com/"
        openai.api_version = "2023-03-15-preview"
        openai.api_key = os.getenv("OPENAI_API_KEY")

        self.base_prompt =  [
            {"role":"system","content":"You are an AI that creates HTML documents with content that matches the URL path and adds relevant href links to the content."},
            {"role":"user","content":"/actors/top10"},
            {"role":"assistant","content":"<!DOCTYPE html>\n<html>\n<head>\n\t<title>Top 10 Actors</title>\n</head>\n<body>\n\t<h1>Top 10 Actors</h1>\n\t<p>Here are the top 10 actors of all time:</p>\n\t<ol>\n\t\t<li><a href=\"/actors/marlon-brando\">Marlon Brando</a></li>\n\t\t<li><a href=\"/actors/al-pacino\">Al Pacino</a></li>\n\t\t<li><a href=\"/actors/robert-de-niro\">Robert De Niro</a></li>\n\t\t<li><a href=\"/actors/meryl-streep\">Meryl Streep</a></li>\n\t\t<li><a href=\"/actors/tom-hanks\">Tom Hanks</a></li>\n\t\t<li><a href=\"/actors/denzel-washington\">Denzel Washington</a></li>\n\t\t<li><a href=\"/actors/jack-nicholson\">Jack Nicholson</a></li>\n\t\t<li><a href=\"/actors/dustin-hoffman\">Dustin Hoffman</a></li>\n\t\t<li><a href=\"/actors/leonardo-dicaprio\">Leonardo DiCaprio</a></li>\n\t\t<li><a href=\"/actors/anthony-hopkins\">Anthony Hopkins</a></li>\n\t</ol>\n</body>\n</html> \n\nThis HTML document displays the top 10 actors of all time."},
            {"role":"user","content":"/incredible/worldplaces"},
            {"role":"assistant","content":"<!DOCTYPE html>\n<html>\n<head>\n\t<title>Incredible World Places</title>\n</head>\n<body>\n\t<h1>Incredible World Places</h1>\n\t<p>There are many incredible places in the world that are worth visiting. Here are some of them:</p>\n\t<ul>\n\t\t<li><a href=\"/incredible/worldplaces/machu-picchu\">Machu Picchu</a></li>\n\t\t<li><a href=\"/incredible/worldplaces/great-wall-of-china\">Great Wall of China</a></li>\n\t\t<li><a href=\"/incredible/worldplaces/pyramids-of-giza\">Pyramids of Giza</a></li>\n\t\t<li><a href=\"/incredible/worldplaces/taj-mahal\">Taj Mahal</a></li>\n\t\t<li><a href=\"/incredible/worldplaces/colosseum\">Colosseum</a></li>\n\t\t<li><a href=\"/incredible/worldplaces/petra\">Petra</a></li>\n\t\t<li><a href=\"/incredible/worldplaces/mount-everest\">Mount Everest</a></li>\n\t\t<li><a href=\"/incredible/worldplaces/niagara-falls\">Niagara Falls</a></li>\n\t\t<li><a href=\"/incredible/worldplaces/grand-canyon\">Grand Canyon</a></li>\n\t\t<li><a href=\"/incredible/worldplaces/santorini\">Santorini</a></li>\n\t</ul>\n</body>\n</html> \n\nThis HTML document displays a list of incredible world places that are worth visiting."}
        ]

        self.current_prompt = self.base_prompt

    def get_response(self, path: str):
        content = f"/{path}"
        self.current_prompt += [{"role":"user","content":content}]

        response = openai.ChatCompletion.create(
        engine="poc-GPT35",
        messages = self.current_prompt,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

        content = response.get('choices')[0].get('message').get('content')
        self.current_prompt += [{"role":"assistant","content":content}]

        return content