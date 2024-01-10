from langchain.agents import Tool
from langchain.tools import BaseTool
import praw
import os
from dotenv import load_dotenv
load_dotenv()
'''
## Maybe add filter to filter by specific interest in a subreddit
reddit.subreddit("all").filters.add("test")

## Add function to search subreddits containing keyword


'''


class Subreddit_Hot_N_Posts(BaseTool):
    name = "Subreddit N HOT Posts"
    description = "Hot Posts Search: Use this when you need to create something based on a number \"n\" current Hottest posts in a specified subreddit.\
              Should be a comma separated list of a String representing the creation request, a String representing the subreddit,\
                  and a integer n represeting the number of posts. For example: `body,test,10` if you are looking to create the text body for a post\
                      based on 10 Hot posts from the test subreddit."
        
    '''
    Any intialization that needs to be done such as auth
    '''
    reddit = praw.Reddit(
        client_id = os.getenv("REDDIT_CLIENT_ID"),
        client_secret = os.getenv("REDDIT_API_KEY"),
        user_agent = os.getenv("REDDIT_API_USER_AGENT"),
    )

    '''
    Retrieves Titles for N "Hot" posts in queried subreddit
    '''
    def _run(self, query: str) -> str:
        req, sub, n = query.split(",")
        n = int(n)
        resp = []
        subreddit = self.reddit.subreddit(sub)
        if req == "post":
            for submission in subreddit.hot(limit=n):
                if submission.stickied:
                    continue
                if submission.selftext != '':
                    post = "Post Title: " + submission.title+\
                        " Post Body: " + submission.selftext
                    resp.append(post)
                else:
                    post = "Post Title: " + submission.title+\
                        " Post Body: " + submission.url
                    resp.append(post)
        elif req == "title":
            for submission in subreddit.hot(limit=n):
             if submission.stickied:
                    continue
             resp.append(submission.title)
        elif req == "body":
            for submission in subreddit.hot(limit=10):
                if submission.stickied:
                    continue
                if submission.selftext != '':
                    resp.append(submission.selftext)
                else:
                    resp.append(submission.url)

        return " ".join(resp)
    
    async def _arun(self, query: str, n: int=10) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Subreddit_Hot_N_Posts does not support async")

class Subreddit_Top_N_Posts(BaseTool):
    '''
    Any intialization that needs to be done such as auth
    '''
    name = "Subreddit N TOP Posts"
    description = "Top Posts Search: Use this when you need to get \"n\" number of Top posts of all time in a specified subreddit.\
              Should be a comma separated list of a String representing the creation request (title or body), a String representing the subreddit,\
                  and a integer n represeting the number of posts. For example: `body,test,10` if you are looking to create the text body for a post\
                      based on 10 Top posts from the test subreddit."
        
    reddit = praw.Reddit(
        client_id = os.getenv("REDDIT_CLIENT_ID"),
        client_secret = os.getenv("REDDIT_API_KEY"),
        user_agent = os.getenv("REDDIT_API_USER_AGENT"),
    )

    '''
    Retrieves Titles for N "TOP" posts in queried subreddit
    '''
    def _run(self, query: str) -> str:
        req, sub, n = query.split(",")
        n = int(n)
        resp = []
        subreddit = self.reddit.subreddit(sub)
        if req == "post":
            for submission in subreddit.top(time_filter="all", limit=n):
                if submission.stickied:
                    continue
                if submission.selftext != '':
                    post = "Post Title: " + submission.title+\
                        " Post Body: " + submission.selftext
                    resp.append(post)
                else:
                    post = "Post Title: " + submission.title+\
                        " Post Body: " + submission.url
                    resp.append(post)
        elif req == "title":
            for submission in subreddit.top(time_filter="all", limit=n):
                if submission.stickied:
                    continue
                title = submission.title
                resp.append(title)
        elif req == "body":
            for submission in subreddit.top(time_filter="all", limit=10):
                if submission.stickied:
                    continue
                if submission.selftext != '':
                    resp.append(submission.selftext)
                else:
                    resp.append(submission.url)

        return ",".join(resp)
    
    async def _arun(self, query: str, n: int=10) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Subreddit_Top_N_Posts does not support async")

class Subreddit_Search_Relevant_N_Posts(BaseTool):
    name = "Subreddit N Relevant Posts By Topic"
    description = "Relevant Posts Search: Use this when you need to create something based on a specific topic based on a number of \"n\" relevant posts in a specified subreddit.\
              Should be a comma separated list of a String representing the creation request, String representing the topic, a String representing the subreddit,\
                  and a integer n represeting the number of posts. For example: `body,interesting stuff,test,10` if you are looking to create the text body for a post\
                      based on 10 Top posts on the topic interesting stuff from the test subreddit."
        
    '''
    Any intialization that needs to be done such as auth
    '''
    reddit = praw.Reddit(
        client_id = os.getenv("REDDIT_CLIENT_ID"),
        client_secret = os.getenv("REDDIT_API_KEY"),
        user_agent = os.getenv("REDDIT_API_USER_AGENT"),
    )

    '''
    Retrieves Titles for N "TOP" posts in queried subreddit
    '''
    def _run(self, query: str) -> str:
        req, topic, sub, n = query.split(",")
        n = int(n)
        resp = []
        subreddit = self.reddit.subreddit(sub)
        if req == "post":
            for submission in subreddit.search(query=topic, sort="relevance", time_filter="all", limit=n):
                if submission.stickied:
                    continue
                if submission.selftext != '':
                    post = "Post Title: " + submission.title+\
                        " Post Body: " + submission.selftext
                    resp.append(post)
                else:
                    post = "Post Title: " + submission.title+\
                        " Post Body: " + submission.url
                    resp.append(post)
        elif req == "title":
            for submission in subreddit.search(query=topic, sort="relevance", time_filter="all", limit=n):
                if submission.stickied:
                    continue
                title = submission.title
                resp.append(title)
        elif req == "body":
            for submission in subreddit.search(query=topic, sort="relevance", time_filter="all", limit=n):
                if submission.stickied:
                    continue
                if submission.selftext != '':
                    resp.append(submission.selftext)
                else:
                    resp.append(submission.url)

        return ",".join(resp)
    
    async def _arun(self, query: str, n: int=10) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Subreddit_Top_N_Posts does not support async")

class RedditPostTool(BaseTool):
    name = "Reddit_Post"
    description = ""
    '''
    Any intialization that needs to be done such as auth
    '''
    reddit = praw.Reddit(
        client_id = os.getenv("REDDIT_CLIENT_ID"),
        client_secret = os.getenv("REDDIT_API_KEY"),
        user_agent = os.getenv("REDDIT_API_USER_AGENT"),
        username = os.getenv("REDDIT_API_ACC_USERNAME"),
        password = os.getenv("REDDIT_API_ACC_PASSWORD"),
    )

    '''
    Runs the custom tool
    '''
    def _run(self, query: str) -> str:

        return
    
         

