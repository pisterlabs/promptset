import pandas as pd
import mysql.connector
from dotenv import load_dotenv
load_dotenv()
import os
import openai

print("Good morning")
print(os.getcwd())

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_translation_and_hint(word_eng, pronunciation):
    prompt = f"You are part of a software. The user will give you an english word and a korean pronounciation. Generate the korean translation in korean letters given these two inputs. Additionally generate a memory hook on how a user could remember the korean word easier. For numbers, please also translate them to the correctly written Korean word (don't just write the number). You then respond in the following format: <translation> \n <memory hook>"
    
    response = openai.ChatCompletion.create(
      model="ft:gpt-3.5-turbo-0613:personal::84rVSAeT",
      messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"English word: {word_eng} - Korean pronounciation: {pronunciation}"}],
    )
    print(response.choices[0].message.content)
    result = response.choices[0].message.content.split("\n")
    
    # Assuming the API returns translation first and hint second

    if (len(result) == 2):
        return result[0].strip(), result[1].strip()
    else:
        return result[0].strip(), ""
    
        
    

# Read the Excel file
df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lesson2.xls'))

# Update dataframe with translations and hints
df["word_kor"], df["hint"] = zip(*df.apply(lambda x: get_translation_and_hint(x["word_eng"], x["pronunciation"]), axis=1))


# Connect to MySQL
conn = mysql.connector.connect(
    host= os.getenv("DB_HOST"),
    user=os.getenv("DB_USERNAME"),
    password= os.getenv("DB_PASSWORD"),
    db= os.getenv("DB_NAME"),
    autocommit = True,
    ssl_ca="/etc/ssl/cert.pem"

)
cursor = conn.cursor()


# Insert data into the table
for _, row in df.iterrows():
    sql = "INSERT INTO words (word_eng, pronunciation, word_kor, hint, level, lesson_id) VALUES (%s, %s, %s, %s, %s, %s)"
    cursor.execute(sql, tuple(row))

# Commit and close connection
conn.commit()
cursor.close()
conn.close()
