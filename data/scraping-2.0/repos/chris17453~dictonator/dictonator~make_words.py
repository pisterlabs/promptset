import os
import json
import openai
import sqlite3
from dictionary_entry import DictionaryEntry
from terms import computer_terms

# Retrieve the API Key from environment variables
api_key = os.getenv('dict_openai_api_secret')

if api_key is None:
    raise ValueError("API_KEY is not set in the environment variables")

# Set up OpenAI with the API Key
openai.api_key = api_key


# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('dictionary_entries.db')

# Create a new SQLite cursor
cur = conn.cursor()

# Create table
cur.execute('''CREATE TABLE IF NOT EXISTS dictionary_entries
            (word TEXT, base_word TEXT, definition TEXT, word_type TEXT, pronunciation TEXT, example TEXT, etymology TEXT, synonyms TEXT, antonyms TEXT, usage_notes TEXT, frequency TEXT, language TEXT, requested_language TEXT, prompt TEXT,image_scene_prompt TEXT)''')

# Commit the changes
conn.commit()

def generate_sutuation(word:str):
    try:
        prompt =f"I want you to generate a feeling. The feeling can cover anything and be based on feeling one state or another. It is a short sentance. Anything Technology or code related. I want it to be based on this word and make it tech related it must contain more than 4 sylables: {word}"
        while True:
            response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=0.6,
                    max_tokens=500,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )

            content = response.choices[0].text.strip()
            if "{prompt}" not in content:
                break

        print("Returning Pre-Prompt")
        return content
    
    except openai.error.OpenAIError as e:
        print(f"Error: {e}")
        return None
    

def generate_word_entry(word,prompt,language="english"):
    try:
        expected_fields = ["word", "definition", "word_type", "example", 
                       "synonyms", "antonyms", "etymology", "pronunciation", 
                       "frequency", "usage_notes", "image_scene_prompt"]

        #prompt =# f"""You are a Lexicographer. You are going to create a new word, a dystopian mashup the. The word should not be simply two words concatenated, but a fun new word. Do not use any words containes in the situation. ONLY give me the json output, nothing else. Create a dictionary entry for the sutuation "{situation}"  in {language} in JSON format including the following fields: word, definition, word_type, example, synonyms, antonyms, etymology, pronunciation, frequency, language, usage_notes."""
        prompt_text="""You are a creative Lexicographer for a new publication that is enhancing the language. Create a new word. 
        This dictionary is text only and cannot show images or urls.
        ONLY give me the json output, nothing else. Create a dictionary entry for the prompt "{0}"  in {1}. 
        Use this json format: {2}""".format(prompt,language,", ".join(expected_fields))
        
        print(word)


        while True:
            

            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt_text,
                temperature=0.6,
                max_tokens=500,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            content = response.choices[0].text.strip()
            # Check for undesired patterns in the content.
            # For example, check for "{prompt}" and URLs.
            print("Checking")
            if "{prompt}" not in content and "http" not in content and "{language}" not in content:
                try:
                    print("LS")
                    json_obj = json.loads(content)
                    # Check if all expected fields are present, and they are not null or empty strings
                    if all(field in json_obj 
                           and len(json_obj['image_scene_prompt'])>10  
                           and isinstance(json_obj['synonyms'],list) 
                           and isinstance(json_obj['antonyms'],list) 
                           and json_obj[field] not in [None, ""] for field in expected_fields):
                        print("JS")
                        json_obj['base_word'] = word
                        json_obj['prompt'] = prompt
                        json_obj['language'] = language
                        json_obj['requested_language'] = language
                        print("Good")
                        return json_obj
                    print("IDK")

                except json.JSONDecodeError:
                    print ("BAD CONTENT, Missing fields")
                print("TRY")
            else:
                print ("BAD CONTENT REDO, data is invalid")
            print("What")
            print(content)
                    
    except openai.error.OpenAIError as e:
        print(f"Error: {e}")
        return None
    print("A normal Exit")

def lookup_word(word):
    # Safeguard against SQL injection by using ? placeholder
    cur.execute('''SELECT * FROM dictionary_entries WHERE word = ?''', (word,))

    # Fetch the result
    result = cur.fetchone()

    # Check if a result was found
    if result:
        # Map the result to a dictionary for better readability
        columns = ['word', 'base_word', 'definition', 'word_type', 'pronunciation', 'example', 'etymology', 'synonyms', 'antonyms', 'usage_notes', 'frequency', 'language', 'requested_language', 'prompt','image_scene_prompt']
        result_dict = {columns[i]: result[i] for i in range(len(columns))}
        
        # Return the resulting dictionary
        return result_dict
    else:
        # Return a message if the word is not found
        return None

# Example Usage
word_to_lookup = "example"  # Replace with the word you want to lookup
print(lookup_word(word_to_lookup))


def update_database(json_obj):
    exists=lookup_word(json_obj['word'])
    if exists is not None:
        print("Duplicate word, skupping")
        return
    json_obj['synonyms'] = ', '.join(json_obj.get('synonyms', []))
    json_obj['antonyms'] = ', '.join(json_obj.get('antonyms', []))

    # Insert data into database
    print ("UPDATING DATABASE")
    print ("UPDATING DATABASE")
    print ("UPDATING DATABASE")
    print ("UPDATING DATABASE")
    try:
        cur.execute('INSERT INTO dictionary_entries VALUES (:word, :base_word, :definition, :word_type, :pronunciation, :example, :etymology, :synonyms, :antonyms, :usage_notes, :frequency, :language, :requested_language, :prompt, :image_scene_prompt)', json_obj)    
        conn.commit()
    except sqlite3.IntegrityError as e:
        print("SQLite integrity error:", e)
        print("Failed data:", json_obj)
    except sqlite3.OperationalError as e:
        print("SQLite operational error:", e)
        print("Failed data:", json_obj)
    except Exception as e:
        print("Unexpected error:", e)
        print("Failed data:", json_obj)


if __name__=="__main__":
    # Example usage:
    for  word in computer_terms:
        prompt = generate_sutuation(word)
        try:
            word_info = generate_word_entry(word,prompt)
            print(word_info)
            print("")
            update_database(word_info)
            #exit()
        except Exception as ex:
            print(ex)
            pass
    conn.close()

