import requests
from pydantic import BaseModel, Field
from langchain.tools import tool




class TwitterInteractionInput(BaseModel):
    message: str = Field(..., description="Message to post on Twitter")

# Replace 'YOUR_API_KEY', 'YOUR_API_SECRET', 'YOUR_ACCESS_TOKEN', and 'YOUR_ACCESS_TOKEN_SECRET' with actual Twitter API credentials
TWITTER_API_KEY = 'YOUR_API_KEY'
TWITTER_API_SECRET = 'YOUR_API_SECRET'
TWITTER_ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'
TWITTER_ACCESS_TOKEN_SECRET = 'YOUR_ACCESS_TOKEN_SECRET'

@tool(args_schema=TwitterInteractionInput)
def post_on_twitter(input_data: TwitterInteractionInput) -> str:
    """Post a message on Twitter."""
    base_url = 'https://api.twitter.com/2/tweets'

    headers = {
        'Authorization': f'Bearer {TWITTER_ACCESS_TOKEN}',
    }

    data = {
        'status': input_data.message,
    }

    try:
        response = requests.post(base_url, headers=headers, data=data)
        
        if response.status_code == 200:
            return "Message posted on Twitter successfully."
        else:
            return f"Twitter API Request failed with status code: {response.status_code}"

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
