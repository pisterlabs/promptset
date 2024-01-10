import tweepy
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

class TwitterBot:
    def __init__(self, twitter_credentials, openai_api_key):
        self.llm = ChatOpenAI(temperature=.7, max_tokens=45, openai_api_key=openai_api_key, model_name='gpt-3.5-turbo')
        self.twitter_credentials = twitter_credentials
        self.openai_api_key = openai_api_key
        self.auth_twitter()
        self.configure_openai_api()

    def auth_twitter(self):
        self.client = tweepy.Client(self.twitter_credentials['TWITTER_BEAR_TOKEN'] , self.twitter_credentials['consumer_key'], self.twitter_credentials['consumer_secret'],self.twitter_credentials['access_token'], self.twitter_credentials['access_token_secret'])
        auth = tweepy.OAuthHandler(self.twitter_credentials['consumer_key'], self.twitter_credentials['consumer_secret'])
        auth.set_access_token(self.twitter_credentials['access_token'], self.twitter_credentials['access_token_secret'])
        self.twitter_api = tweepy.API(auth)

    def configure_openai_api(self):
        openai.api_key = self.openai_api_key
        
    def reply_to_mentions(self):
        mentions = self.twitter_api.mentions_timeline()

        for mention in mentions:
            user_text = mention.text
            reply_text = self.generate_reply(user_text)
            reply_to_tweet_id = mention.id

            try:
                self.twitter_api.update_status(f"@{mention.user.screen_name} {reply_text}", in_reply_to_status_id=reply_to_tweet_id)
                print(f'Replied to tweet by @{mention.user.screen_name}: {mention.text}')
            except tweepy.TweepError as e:
                print(f"Error replying to tweet: {str(e)}")

    def generate_reply(self, mentioned_conversation_tweet_text):
        system_template = """
            You are a creative genius with a knack for innovative ideas.
            Your goal is to inspire and guide users with their creative projects.
            
            % RESPONSE TONE:

            - Your responses should be enthusiastic, encouraging, and imaginative
            - Inject a sense of curiosity and wonder into your tone
            
            % RESPONSE FORMAT:

            - Share inspiring ideas concisely
            - Keep responses engaging, but not too lengthy
            - Avoid the use of emojis
            
            % RESPONSE CONTENT:

            - Offer unique and imaginative suggestions
            - If you need more information, kindly ask the user for details
            - If an idea isn't coming to mind, express curiosity and encourage exploration
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        # get a chat completion from the formatted messages
        final_prompt = chat_prompt.format_prompt(text=mentioned_conversation_tweet_text).to_messages()
        response = self.llm(final_prompt).content
        response = response.replace('"', '')
        
        return response

    def reply_to_mention_by_conversation_id(self, conversation_id, mentioned_conversation_tweet_text):
        try:
            respone = self.generate_reply(mentioned_conversation_tweet_text)
            
            respone = respone.replace('\n', ' ')  # Replace newline with space

            print(respone)
            self.client.create_tweet(in_reply_to_tweet_id=conversation_id, text=respone)
            print("Reply sent successfully.")
        except tweepy.errors.TweepyException as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Replace with your actual Twitter and OpenAI API credentials
    twitter_credentials = {
        'consumer_key': '',
        'consumer_secret': '',
        'access_token': '-',
        'access_token_secret': '',
        'TWITTER_BEAR_TOKEN' : ''
    }

    openai_api_key = ''

    bot = TwitterBot(twitter_credentials, openai_api_key)
    conv_id = 1706707922867069205
    conv_text = "Top 5 Memorable Opening Performances in Olympic History! 🎶🔥 From iconic anthems to breathtaking choreography, relive the most electrifying moments that set the stage for the #Olympics2024. Get ready to be amazed! @lwtprettylaugh"
    bot.reply_to_mention_by_conversation_id(conv_id, conv_text)
