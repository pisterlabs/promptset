from langchain.tools import YouTubeSearchTool
from ast import literal_eval

youtube = YouTubeSearchTool()

def youtube_response(prompt):
    result = youtube(prompt)

    return "The trailer link is " + literal_eval(result)[0]