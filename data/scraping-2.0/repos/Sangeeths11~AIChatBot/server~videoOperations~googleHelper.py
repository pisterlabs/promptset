import openai
import os
from googleapiclient.discovery import build
from textblob import TextBlob
import json
#import appconfig as config
from pytube import YouTube
from dotenv import load_dotenv
import api.appconfig as config
load_dotenv()
#os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY


from google.oauth2 import service_account

# Load the credentials from the JSON file
credentials = service_account.Credentials.from_service_account_file(config.CREDENTIALS_PATH, scopes=['https://www.googleapis.com/auth/youtube.force-ssl'])


#youtube = build("youtube", "v3", developerKey=os.getenv("GOOGLE_API_KEY"))
youtube = build("youtube", "v3", credentials=credentials)


def youtubeSearch(query, count=2):
    """
    Searches YouTube for videos matching a given query.

    Args:
      query (str): The query to search for.
      count (int): The maximum number of results to return.

    Returns:
      list: A list of dictionaries containing the video title, watch URL, and sentiment score.

    Examples:
      >>> youtubeSearch("cat videos", count=3)
      [{'name': 'Cats Being Jerks Video Compilation', 'watchUrl': 'https://www.youtube.com/watch?v=VHXmC2eY7t8', 'sentimentScore': -2}, 
      {'name': 'Cats vs. Zombies - A Halloween Special', 'watchUrl': 'https://www.youtube.com/watch?v=VHXmC2eY7t8', 'sentimentScore': 3}, 
      {'name': 'Cats vs. Dogs: Who Wins?', 'watchUrl': 'https://www.youtube.com/watch?v=VHXmC2eY7t8', 'sentimentScore': 1}]
    """
    response = youtube.search().list(
        q=query,
        type="video",
        part="id,snippet",  
        maxResults=count
    ).execute()
    
    results = []
    for result in response.get("items", []):
        if result["id"]["kind"] == "youtube#video":
            videoId = result["id"]["videoId"]
            videoTitle = result["snippet"]["title"]
            
            url = f"https://www.youtube.com/watch?v={videoId}"
            

            sentimentScore = getSentimentOfVideo(videoId)
            if sentimentScore:
                results.append({"name": videoTitle, "watchUrl": url, "sentimentScore": sentimentScore})
            else:
                continue
    return results    
    
    

def getSentimentOfVideo(videoId, commentCount = 12):
    """
    Gets the sentiment score of a given YouTube video.

    Args:
      videoId (str): The ID of the YouTube video.
      commentCount (int): The maximum number of comments to consider.

    Returns:
      int: The sentiment score of the video.

    Examples:
      >>> getSentimentOfVideo("VHXmC2eY7t8", commentCount=10)
      -2
    """
    commentList = getCommentsOfVideo(videoId, commentCount)
    if commentList:
        return getSentimentOfComments(commentList)
    else:
        return None
    

def getCommentsOfVideo(videoId, commentCount):
    """
    Gets the comments of a given YouTube video.

    Args:
      videoId (str): The ID of the YouTube video.
      commentCount (int): The maximum number of comments to return.

    Returns:
      list: A list of strings containing the comments of the video.

    Examples:
      >>> getCommentsOfVideo("VHXmC2eY7t8", commentCount=3)
      ["This video is so funny!", "I love cats!", "Cats are the best!"]
    """
    try:
        comments = youtube.commentThreads().list(
            part="snippet",
            videoId=videoId,
            textFormat="plainText",
        ).execute()
    
        commentList = []
        for comment in comments["items"][:min(commentCount, comments["pageInfo"]["totalResults"], comments["pageInfo"]["resultsPerPage"])]:
            commentText = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            commentList.append(commentText)
            
        return commentList
    except:
        print(f"Error while retriving comments of video with videoId: {videoId}, video excluded.")   
        return None

    
def getSentimentOfComments(commentList):
    """
    Gets the sentiment score of a list of comments.

    Args:
      commentList (list): A list of strings containing the comments.

    Returns:
      int: The sentiment score of the comments.

    Examples:
      >>> getSentimentOfComments(["This video is so funny!", "I love cats!", "Cats are the best!"])
      3
    """
    positiveCount = 0
    negativeCount = 0

    for commentText in commentList:
        analysis = TextBlob(commentText)
        sentimentScore = analysis.sentiment.polarity

        if sentimentScore > 0:
            positiveCount += 1
        elif sentimentScore < 0:
            negativeCount += 1

    return positiveCount - negativeCount


def getEmbedUrl(video):
    """
    Gets the embed URL of a given YouTube video.

    Args:
      video (dict): A dictionary containing the video title, watch URL, and sentiment score.

    Returns:
      str: The embed URL of the video.

    Examples:
      >>> getEmbedUrl({'name': 'Cats Being Jerks Video Compilation', 'watchUrl': 'https://www.youtube.com/watch?v=VHXmC2eY7t8', 'sentimentScore': -2})
      "https://www.youtube.com/embed/VHXmC2eY7t8"
    """
    url = video.get("watchUrl", "")
    yt = YouTube(url)
    return yt.embed_url
    