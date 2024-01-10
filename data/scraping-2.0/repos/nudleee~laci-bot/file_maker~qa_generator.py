import pandas as pd
import os
import openai

openai.api_key  = os.environ['OPENAI_API_KEY']

raw = pd.read_csv("~/Downloads/SzakmaiGyakorlatQnA.csv", encoding='latin-1')

emails = pd.DataFrame(columns=['Levelezés'])

for index, row in raw.iterrows():
    email = row[0].split("\n")
    stripped_emails = [string.strip() for string in email]
    emails = pd.concat([emails, pd.DataFrame([['\n'.join(stripped_emails)]], columns=['Levelezés'])], ignore_index=True)

emails.to_csv('files/formatted_emails.csv', index=False)

def generate_faq(email):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
        {"role": "system", "content": "Egy FAQ generátor vagy"},
        {"role": "user", "content": f"""
        Generálj néhány kérdés-válasz párt az alábbi email váltásból. Ügyelj, hogy az eredmény ne tartalmazzön érzékeny információt,
        azaz ne legyen benne név, cégnév, emailcím. A kérdés-válasz párokat a BME VIK szakmai gyakolattal kapcsolatban kell létrehozni.
        A válasz csv formátumnak feleljen meg.
        
        Email:
        {email}

        Példa:
        "Mikor kell felvenni a Neptunban a Szakmai gyakorlat tárgyat?";"Amennyiben a gyakorlatot az ajánlott módon a nyári időszakban végzed, akkor a tárgyat a következő őszi félévben kell felvenni. Ha a szakmai gyakorlatot az ajánlottól eltérő módon félév közben végzed, akkor ugyanabban a félévben. A szakmai gyakorlat teljesítése a végbizonyítvány (abszolutórium) kiállításának feltétele, ezért fontos, hogy időben meglegyen belőle az aláírásod."
        """
        },
        ],
        temperature=0.7,
        max_tokens = 500
    )
    return response['choices'][0]['message']['content'].split("\n")

generated_qa = pd.DataFrame(['Kérdés', 'Válasz'])

for index, row in emails.iterrows():
    email = row['Levelezés']
    if index % 10 == 0:
        print(f"{index}/{len(emails)}")
    try:    
        response = generate_faq(email)
    except Exception as e :
        print(e)

    data = []
    for r in response:
        try:
            qna = r.split(';')
            question = qna[0].strip('"')
            answer = qna[1].strip('"')
            data.append([question, answer])
        except Exception as e :
            print(e)
            
    generated_qa = pd.concat([generated_qa, pd.DataFrame(data, columns=['Kérdés', 'Válasz'])], ignore_index=True)

# generated_qa = pd.read_csv("files/generated.csv", delimiter=',')
generated_qa.to_csv("files/fix_generated.csv", columns=["Kérdés", "Válasz"],index=False)

