import openai
import json

from dotenv import load_dotenv
import os

load_dotenv()

secret_key = os.environ.get('SECRET_KEY')
class apiParser:
    
    def testApi(search_query):
        
       
        prompt=f"""You are a sentiment analyzer.You will be given an input from a user 
        describing their mood ,and you will give a score for the songs that the user might prefer.
        You will give me output as a map(key:value) pairs where key is the 'category' and value is the 'score'.
        You will output ONLY a score between 0 and 1 (two decimal places) for the following categories:
        1. Valence
        2. Danceability
        3. Energy
        4. Tempo
       
        
        Here is the search query: {search_query}
        """
        openai.api_key =secret_key
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print(response.choices[0].message.content)
        json_API = json.loads(response.choices[0].message.content)
        return json_API
       