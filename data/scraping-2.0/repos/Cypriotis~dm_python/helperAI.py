import openai
import random
from db_connect import DatabaseConnector
from lastfm_api import LastFmAPI


# MySQL database connection details
host = 'localhost'
username = 'root'
password = ''
database = 'DeMa'
port = '3308'

# Set up OpenAI API credentials
openai.api_key = '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'


# Last.fm API key
api_key = '899c27976ecfa7b4adf9f276b445d62f'

# Create an instance of the LastFmAPI
lastfm_api = LastFmAPI(api_key)

# Create an instance of the DatabaseConnector
db_connector = DatabaseConnector(host, username, password, database, port)
db_connector.connect()

class chatgpt:
        def execute():

            # Generate random first names using ChatGPT
            def generate_first_names():
                response = openai.Completion.create(
                    engine='text-davinci-003',
                    prompt='Generate a random first name:',
                    max_tokens=10,
                    n=2,  # Number of names to generate
                    stop=None,
                    temperature=0.6     
                )
                print("generated")

                first_names = [choice['text'].strip() for choice in response['choices']]
                return first_names

            # Test the first name generation
            random_first_names = generate_first_names()

            # Print the generated first names
            for index, name in enumerate(random_first_names, start=1):
                #print(f"First Name {index}: {name}")
                user_top_tracks = lastfm_api.get_user_top_songs(name)
                tracks = user_top_tracks['lovedtracks']['track']
                songs_amount = tracks['lovedtracks']['total']
                print(songs_amount)
                #name = user_info['user']['name']
                query = f"INSERT INTO Test (name) VALUES ('{name}')"
                db_connector.execute_query(query)
                db_connector.commit_changes() 



                

