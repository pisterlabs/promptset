import tiktoken
import openai
import json

class AutoDescription:
    def __init__(self, openai_key, tables):
        self.tables = tables
        # self.initial_description, self.description_PQ, self.potential_query = self.generate_description(self.context)
        openai.api_key = (openai_key)
    
    def generate(self):
        descriptions = {}
        i = 0
        for key, value in self.tables.items():
            
            i+=1
            print(i)
            
            context = json.dumps(value, indent=4)
            num_tokens = self.num_tokens_from_string(context)
            if num_tokens > 4000:
                context = context[:4000]
            description = self.generate_description(context, "gpt-3.5-turbo")
            descriptions[key] = description
            
            if i % 100 == 0:
                with open('descriptions'+str(i)+'.json', 'w') as f:
                    json.dump(descriptions, f)
            
        with open('descriptions.json', 'w') as f:
            json.dump(descriptions, f)
            
    def num_tokens_from_string(self, string, encoding_name="gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
        
    def generate_description(self, context, model):
        description = openai.ChatCompletion.create(
            model=model,
            messages=[
                    {
                        "role": "system", 
                        "content": "You are an assistant for a dataset search engine.\
                                    Your goal is to increase the performance of this dataset search engine for keyword queries."},
                    {
                        "role": "user", 
                        "content": """Instruction:
Answer the questions while using the input and context.
The input is a table in json format.

Input:
""" + context + """
Question:
Describe the table in one complete and coherent paragraph.

Answer: """},
                ],
            temperature=0.3)
        description_content = description.choices[0]['message']['content']
        return description_content

def main():
    def read_tables(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    OPENAI_API_KEY = "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"
    table_path = './wikitables/tables.json'
    tables = read_tables(table_path)
    generator = AutoDescription(OPENAI_API_KEY, tables)
    generator.generate()

if __name__ == "__main__":
    main()