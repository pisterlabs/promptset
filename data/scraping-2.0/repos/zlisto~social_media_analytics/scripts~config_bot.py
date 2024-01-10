import openai

# Credentials
api_key = ''
api_secret = ''
access_key = ''
access_secret = ''
credentials = {'consumer_key': api_key, 'consumer_secret': api_secret}
user_id = ''
OPENAI_API_KEY = ''

#OPENAI 
openai.api_key = OPENAI_API_KEY

# Twitter URLs
urls = {'authorize': "https://api.twitter.com/oauth/authorize",
        'access_token': "https://api.twitter.com/oauth/access_token",
        'tweets': "https://api.twitter.com/2/tweets",
        'retweets': "https://api.twitter.com/2/users/{user_id}/retweets",
        'media': 'https://upload.twitter.com/1.1/media/upload.json',
        'like': "https://api.twitter.com/2/users/{user_id}/likes",
        'follow': "https://api.twitter.com/1.1/friendships/create.json",
        'bio': "https://api.twitter.com/1.1/account/update_profile.json",
        'request_token': "https://api.twitter.com/oauth/request_token?oauth_callback=oob&x_auth_access_type=write"
       }
