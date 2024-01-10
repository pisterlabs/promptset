import openai

class Convert_to_sql:
    
    def __init__(self,api_path='api_key.txt'):
        openai.api_key = self.open_file(api_path)
        self.base_prompt = "### Postgres SQL table, with their properties:\n#\n# movies(id,name,genre, year_of_production,best_imdb_score,duration)\n#\n### A query to return : "
        self.base_prompt2 = "\nSELECT"
        
        
    def open_file(self,filepath):
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()
        
    def generate_response(self, user_query):

        response = openai.Completion.create(
        model="text-davinci-003",
        prompt= self.base_prompt + user_query + self.base_prompt2,
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["#", ";"]
        )
        
        return 'SELECT'+response['choices'][0]['text']

