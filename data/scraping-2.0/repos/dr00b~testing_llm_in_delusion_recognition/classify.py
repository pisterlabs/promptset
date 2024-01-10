"""
Create a class called Classifier.
In the init function, load environment variables and save them to self.
Read davinci_base_prompt.txt and save to self.base_prompt.
"""

import dotenv
import os
import openai
import sqlite3
dotenv.load_dotenv()

class Classifier:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.database_path = os.path.join("data", os.getenv('SQLITE_DB_NAME'))       
        self.prompt_version = int(os.getenv('PROMPT_VERSION'))
        self.base_prompt = open(os.path.join("prompts", f'davinci_base_prompt_v{self.prompt_version}.txt'), 'r').read()
        self.create_classification_table()
        self.parsing_error_count = 0

    def classify_text(self, text, text_id=None):
        """
        Use the OpenAI API davinci model to classify the text.
        """
        # TODO: consider top_p to see alternatives considered
        # TODO: tinker w/ frequency_penalty / precense penalty

        openai.api_key = self.api_key
        prompt = self.base_prompt.replace("{text}", text)
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            top_p=1,
            temperature=0,
            max_tokens=500,
            frequency_penalty=0,
            presence_penalty=0,
        )
        if len(response.choices[0].text) == 0:
            raise("Empty response from OpenAI, possibly due to stop words")
        return response, text_id

    def extract_openapi_response(self, response, input_text_id=None):
        response_text = response.choices[0].text
        possible_delusion = None
        excerpt = None
        dominant_theme = None
        parsing_error = False
        try: 
            if self.prompt_version == 2:
                possible_delusion = True if response_text.split("Possible Delusion: ")[1].split("\n")[0] == "true" else False
                excerpt = response_text.split("Excerpt: ")[1].split("\n")[0]
                dominant_theme = response_text.split("Dominant Theme: ")[1].split("\n")[0]
            elif self.prompt_version == 3:
                possible_delusion = True if response_text.split("Possible Delusion: ")[1].split("\n")[0] == "true" else False
                excerpt = response_text.split("Excerpt: ")[1].split("\n")[0]
            elif self.prompt_version == 4:
                dominant_theme = response.choices[0].text
            elif self.prompt_version == 5:
                possible_delusion = True if response_text.split("Possible Delusion: ")[1].split("\n")[0].lower() == "true" else False
                excerpt = response_text.split("Excerpt: ")[1].split("\n")[0]
        except IndexError:
            print(response_text)
            parsing_error = True
            self.parsing_error_count += 1
        response_created = response.created
        id = response.id
        model = response.model
        method = response.object
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        return (input_text_id, response_text, possible_delusion, excerpt, dominant_theme, parsing_error, response_created, id, model, method, prompt_tokens, completion_tokens, total_tokens, self.prompt_version)

    def create_classification_table(self):
        conn = sqlite3.connect(self.database_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS classifications (
            input_text_id INT,
            full_response_text STRING,
            is_possible_delusion BOOLEAN,
            excerpt STRING,
            dominant_theme STRING,
            parsing_error BOOLEAN,
            created_ts INT,
            response_id STRING,
            model STRING,
            object STRING,
            prompt_tokens INT,
            completion_tokens INT,
            total_tokens INT,
            prompt_version INT,
            load_ts DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()
        conn.close()

    def classify_batch_and_save(self, batch_size=100):
        conn = sqlite3.connect(self.database_path)
        c = conn.cursor()
        batch_complete = False
        while not batch_complete:
            c.execute(f"""
            SELECT rowid, comment_text FROM comments
            WHERE to_classify = 1
            AND NOT EXISTS (
                SELECT 1 FROM classifications WHERE input_text_id = comments.rowid
                AND prompt_version = {self.prompt_version}
            )
            LIMIT {batch_size}
            """)
            results = c.fetchall()
            print("Executing Batch Size: ", len(results))
            if len(results) == 0:
                batch_complete = True
                break
            for row in results:
                print("Classifying: ", row)
                text_id, text = row
                response, text_id = self.classify_text(text, text_id)
                response_data = self.extract_openapi_response(response, input_text_id=text_id)
                c.execute("""INSERT INTO classifications 
                (input_text_id, full_response_text, is_possible_delusion, excerpt, dominant_theme, parsing_error, created_ts, response_id, model, object, prompt_tokens, completion_tokens, total_tokens, prompt_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", response_data)
                conn.commit()
                if self.parsing_error_count > 5:
                    raise("Parsing Error Count > 5")


if __name__ == "__main__":
    classifier = Classifier()
    classifier.classify_batch_and_save()