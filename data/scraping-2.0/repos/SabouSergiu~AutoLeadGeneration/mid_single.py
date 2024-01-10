import argparse
def find_owners(df_owner):
    import pandas as pd
    import time
    import openai
    openai.api_key ="API-KEY"
    from unidecode import unidecode
    import numpy as np
    import random
    import re
    import urllib.parse
    from unidecode import unidecode
    from langdetect import detect
    from deep_translator  import GoogleTranslator
    from nameparser import HumanName 
    import string
    import threading
    import concurrent.futures
    import os
    pd.options.mode.chained_assignment = None
    time.sleep(2)
    #=========================================GPT====================================================
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    def similarity(str1, str2):
        """
        Calculates the similarity between two strings using the Levenshtein distance algorithm.
        """
        m, n = len(str1), len(str2)
        # Create a matrix to store the distances between substrings of the two strings
        dist = np.zeros((m+1, n+1))
        # Initialize the first row and column of the matrix
        for i in range(m+1):
            dist[i, 0] = i
        for j in range(n+1):
            dist[0, j] = j
        # Calculate the distances between substrings of the two strings
        for j in range(1, n+1):
            for i in range(1, m+1):
                if str1[i-1] == str2[j-1]:
                    cost = 0
                else:
                    cost = 1
                dist[i, j] = min(dist[i-1, j] + 1,      # deletion
                                dist[i, j-1] + 1,      # insertion
                                dist[i-1, j-1] + cost) # substitution
        # Return the similarity score
        return 1 - dist[m, n] / max(m, n)

    def query_gpt(text,company):
        txt = f"""Who is/are the owner/s,CEO,President,any kind of Manager,any kind of Director,founder,co-founder,creator or other important position in the company of {company} company in 2023 based only on this text: "{text}"
    If the text doesn't provide the name/s of the owner/s,CEO,President,any kind of Manager, any kind of Director,founder,co-founder,creator or other important position write "Not found".
    Tell me only the names separated by a new line and the rank and company separated by a "-" or “Not found” in case the text doesn't provide the information, no extra text. Only name/s of people separated by a new line or "Not found".
    There are four cases:
    1. You have names each with a position within the company or former company -> "{{Name1}} - {{Rank1}} - {{Company1}}\n{{Name2}} - {{Rank2}} - {{Company2}}"
    2. You have names with a position within company or former company and names without a position within the company or former company -> "{{NameWithPosition1}} - {{Rank1}} - {{Company1}}\n{{NameWithPosition2}} - {{Rank2}} - {{Company2}}"
    3. You have names but without a position within the company or former company -> "Not found"
    4. You have no names -> "Not found"
    """
        return txt


    def delete_not_found_lines(string,string_to_delete):
        lines = string.split('\n')
        new_lines = []
        for line in lines:
            if string_to_delete not in line:
                new_lines.append(line)
        return '\n'.join(new_lines)

    def delete_not_company_lines(string,string_to_delete):
        lines = string.split('\n')
        new_lines = []
        ok = 0
        for line in lines:
            ok = 0
            if(line.count(' - ')<2):
                continue
            for string_to in string_to_delete.replace(',',' ').replace('&',' ').split(' '):
                for elem in line.split(' - ')[-1].split(' '):
                    if('wine' not in string_to.lower() and 'vineyard' not in string_to.lower()):
                        if(unidecode(string_to.lower()) == unidecode(elem.lower()) and len(string_to)>2):
                            ok = 1
            if(ok==1):
                new_lines.append(line)
            else:
                if(similarity(line.split(' - ')[-1],string_to_delete)>0.7):
                    new_lines.append(line)
        return '\n'.join(new_lines)
    #=========================================GPT====================================================
    #=========================================LINKEDIN====================================================
    def translate_to_english(text):
        try:
            if detect(text) != 'en':
                return GoogleTranslator(source='auto', target='en').translate(text)
            else:
                return text
        except:
            return text


    def check_linkedin(name_position, linkedin_link):    
        # extract name and position from first line
        for person in name_position.split("\n"):
            ok=0
            for name in remove_special_characters(person.split(' - ')[0]).split(' ')[:3]:
                if(name.lower() in linkedin_link.lower() and name!=''):
                    ok= ok+1
            if(len(remove_special_characters(person.split(' - ')[0]).split(' '))>1):
                if(remove_special_characters(person.split(' - ')[0]).split(' ')[1].lower() in linkedin_link.lower()):
                    ok = ok+1
            if(ok>1):
                #print(person.split(' - ')[0].split(' ')[:3])
                return True
        return False

    def find_linkedin_links(name,text):
        pattern = r"(?i)\b(?:https?://)?(?:www\.)?linkedin\.com/in/\S+\b"
        matches = set(re.findall(pattern, text))
        good_matches = set()
        for match in matches:
            if(check_linkedin(name, urllib.parse.unquote(match)) is True):
                good_matches.add(match)
        
        return list(matches - good_matches), list(good_matches)

    def remove_special_characters(text):
        return re.sub(' +', ' ',re.sub('\".+?\"', '',re.sub(r'\b\w\b', '',re.sub(r'\([^)]*\)', '', re.sub(r'\b\w+\b\.', '',re.sub(r"\b\w+\b\'", '',text))))))

    def find_linkedin(owner,text):
        other_linkedin = []
        owner_linkedin = []
        other_linkedins, owner_linkedins = find_linkedin_links(owner, text)
        if len(owner_linkedins) == 0:
            owner_linkedin.append('---')
        else:
            for k in range(len(owner_linkedins)):
                el = owner_linkedins[k].find('",')
                owner_linkedin.append(owner_linkedins[k][:el])
        if len(other_linkedins) == 0:
            other_linkedin.append('---')
        else:
            for j in range(len(other_linkedins)):
                el1 = other_linkedins[j].find('",')
                if other_linkedins[j].find('linkedin.com/in/') != -1:
                    other_linkedin.append(other_linkedins[j][:el1])
        return owner_linkedin, other_linkedin
    
    df_owner['owner'] = '---'
    df_owner['OwnerLinkedin'] = '---'
    df_owner['OtherLinkedin'] = '---'
    df_owner['gpt_retry'] = 0
    df_owner['name'] = '---'
    df_owner['rank'] = '---'   
    positions = ['owner', 'founder', 'ceo', 'president', 'director', 'proprietor', 'chief', 'partner', 'chairman', 'manager','officer','principal','executive',
                'VP', 'CFO', 'COO', 'CTO', 'CIO', 'CMO', 'CDO', 'CRO', 'CHRO', 'GM',
                'MD', 'ED', 'CLO', 'GC', 'CDO', 'CRO', 'CHRO', 'PM', 'TL', 'CD', 'CPO',
                'CSO', 'CCO', 'CINO', 'CMO', 'CISO', 'CAO', 'CXO', 'CSCO', 'CPO', 'CQO',
                'CTO', 'CTO', 'CRO', 'CCO', 'CKO', 'CLO', 'CTO', 'CRO', 'CBO', 'CE', 'CAO',
                'CeCO', 'CCSO', 'CDO', 'CDO', 'CHRO', 'PM', 'TL', 'CD', 'CPO', 'CSO', 'CCO',
                'CINO', 'CSO', 'CCSO', 'CCMO', 'CDIO', 'CHO', 'CSIO', 'CPSO', 'CTO', 'CNO'
                 ]

    if(df_owner['gpt_data'][0]=='---'):
        return df_owner
    while(df_owner['owner'][0]=='---' and df_owner['gpt_retry'][0]<3):
        text = df_owner['gpt_data'][0]
        company = df_owner['company'][0]
        try:
            completion = openai.ChatCompletion.create(model = "gpt-3.5-turbo",messages=[{"role":"user","content":query_gpt(text,company)}])
            owner = completion.choices[0].message.content
            time.sleep(random.uniform(1.3, 1.5))
        except:
            time.sleep(5)
            continue
        if(owner == 'Not found'):
            owner = '---'
        df_owner['owner'][0] = delete_not_company_lines(delete_not_found_lines(delete_not_found_lines(owner,'Not f'),"I'm sorry"),company)

        if(df_owner['owner'][0]==''):
            df_owner['owner'][0] = '---'
        if df_owner['owner'][0] != '---':
            count20 = 0
            for rund in df_owner.owner[0].split('\n')[:2]:
                valid = 0
                pps = remove_punctuation(translate_to_english(rund.split(' - ')[-2].lower()))
                for lolxas in pps.split(' '):
                    for position in positions:
                        if position.lower() == lolxas:
                            valid = 1
                            break
                    if valid == 1:
                        break
                if(valid == 1):
                    count20=count20+1
                if(count20 == 2):
                    break
            if(count20 == 0):
                df_owner.name[0] = '---'
                df_owner['rank'][0] = '---'
                df_owner['owner'][0] = '---'
            else:
                owner_linkedin, other_linkedin = find_linkedin('\n'.join(df_owner.owner[0].split('\n')[:count20]),df_owner['text_data'][0]) 
                df_owner['OtherLinkedin'][0] = '\n'.join(other_linkedin)
                df_owner['OwnerLinkedin'][0] = '\n'.join(owner_linkedin)
            if(count20 == 1):
                parsed_name = HumanName(df_owner.owner[0].split('\n')[0].split(' - ')[0])
                df_owner.name[0] = remove_punctuation(parsed_name.first + " " + parsed_name.last)
                df_owner['rank'][0] = df_owner.owner[0].split('\n')[0].split(' - ')[-2]
            if(count20 == 2):
                parsed_name = HumanName(df_owner.owner[0].split('\n')[0].split(' - ')[0])
                parsed_name1 = HumanName(df_owner.owner[0].split('\n')[1].split(' - ')[0])
                df_owner.name[0] = remove_punctuation(parsed_name.first + " " + parsed_name.last + "\n" + parsed_name1.first + " " + parsed_name1.last)
                df_owner['rank'][0] = df_owner.owner[0].split('\n')[0].split(' - ')[-2] + "\n" + df_owner.owner[0].split('\n')[1].split(' - ')[-2]
        print('retry:',df_owner['gpt_retry'][0])
        df_owner['gpt_retry'][0] = df_owner['gpt_retry'][0] + 1

    return df_owner


def main(df_owner):
    df_owner = find_owners(df_owner)
    return df_owner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("df_owner")
    args = parser.parse_args()
    main(args.name)