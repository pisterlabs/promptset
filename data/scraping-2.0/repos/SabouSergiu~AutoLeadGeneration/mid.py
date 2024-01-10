import argparse
def find_owners(path):
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
    import json
    semaphore = threading.Semaphore(3)
    #=========================================GPT====================================================
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    def remove_punctuation_withSpace(text):
        return text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
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
                if(name.lower() in linkedin_link[linkedin_link.find('/in/')+4:].lower() and name!=''):
                    ok= ok+1
            if(len(remove_special_characters(person.split(' - ')[0]).split(' '))>1):
                if(remove_special_characters(person.split(' - ')[0]).split(' ')[1].lower() in linkedin_link[linkedin_link.find('/in/')+4:].lower()):
                    ok = ok+1
            if(ok>1):
                print(person.split(' - ')[0].split(' ')[:3])
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
    
    while(os.path.exists(path+'\\start.csv')==False):
        time.sleep(3)
    df_owner = pd.read_csv(path+'\\start.csv')
    df_owner['owner'] = '---'
    df_owner['OwnerLinkedin'] = '---'
    df_owner['OtherLinkedin'] = '---'
    df_owner['gpt_retry'] = 0
    df_owner['name'] = '---'
    df_owner['rank'] = '---'
    df_owner['company_found'] = '---'
    
    
    if(os.path.exists(path+'\\mid.csv')==True):
        print("mid.csv exists")
        df_mirror = pd.read_csv(path+'\\mid.csv')
        if(len(df_mirror)<len(df_owner)):  
            for i in range(len(df_mirror)):
                df_owner['owner'][i] = df_mirror['owner'][i]
                df_owner['OwnerLinkedin'][i] = df_mirror['OwnerLinkedin'][i]
                df_owner['OtherLinkedin'][i] = df_mirror['OtherLinkedin'][i]
                df_owner['gpt_retry'][i] = df_mirror['gpt_retry'][i]
                df_owner['name'][i] = df_mirror['name'][i]
                df_owner['rank'][i] = df_mirror['rank'][i]
        else:
            df_owner = df_mirror

    df_owner.to_csv(path+'\\mid.csv',index=False, encoding = "utf-8-sig")
            
    def check_position(position_to_check):
        positions = ['owner', 'founder', 'ceo', 'president', 'director', 'proprietor', 'chief', 'partner', 'chairman', 'manager','officer','principal','executive',
                    'VP', 'CFO', 'COO', 'CTO', 'CIO', 'CMO', 'CDO', 'CRO', 'CHRO', 'GM',
                    'MD', 'ED', 'CLO', 'GC', 'CDO', 'CRO', 'CHRO', 'PM', 'TL', 'CD', 'CPO',
                    'CSO', 'CCO', 'CINO', 'CMO', 'CISO', 'CAO', 'CXO', 'CSCO', 'CPO', 'CQO',
                    'CTO', 'CTO', 'CRO', 'CCO', 'CKO', 'CLO', 'CTO', 'CRO', 'CBO', 'CE', 'CAO',
                    'CeCO', 'CCSO', 'CDO', 'CDO', 'CHRO', 'PM', 'TL', 'CD', 'CPO', 'CSO', 'CCO',
                    'CINO', 'CSO', 'CCSO', 'CCMO', 'CDIO', 'CHO', 'CSIO', 'CPSO', 'CTO', 'CNO'
                    ]
        for positionx in position_to_check.split(' '):
            for position in positions:
                if(position.lower() == positionx):
                    return True

    
    # declare df_lock as a global variable
    df_lock = threading.Lock()

    def process_row(i):
        if(df_owner['owner'][i]=='---' and df_owner['gpt_data'][i]!='---' and df_owner['gpt_retry'][i]<3):
            while df_owner['gpt_retry'][i] < 3 and df_owner['owner'][i] == '---':
                try:
                    completion = openai.ChatCompletion.create(model = "gpt-3.5-turbo",messages=[{"role":"user","content":query_gpt(df_owner['gpt_data'][i],df_owner['company'][i])}])
                    df_owner['owner'][i] = completion.choices[0].message.content
                    time.sleep(random.uniform(1.3, 1.5))
                    if(df_owner['owner'][i] == 'Not found'):
                        df_owner['owner'][i] = '---'
                    df_owner['owner'][i] = delete_not_company_lines(delete_not_found_lines(delete_not_found_lines(df_owner['owner'][i],'Not f'),"I'm sorry"),df_owner['company'][i])
                    if(df_owner['owner'][i]==''):
                        df_owner['owner'][i] = '---'
                    df_owner['gpt_retry'][i] += 1
                    print(i,". ",df_owner['company'][i], "retrying", df_owner['gpt_retry'][i])
                except Exception as e:
                    with open(path+'\\errorsEncounteredGPT.txt', 'a') as file:
                        file.write(str(e) + '\n at' +str(i)+ '\n\n')
                    df_owner['gpt_retry'][i] += 1
                    print(i,". ",df_owner['company'][i], "retrying", df_owner['gpt_retry'][i])
                    time.sleep(10)
                    
            name_array = []
            rank_array = []
            company_array = []
            print(df_owner['owner'][i])
            if(df_owner['owner'][i]!='---'):
                for j,name in enumerate(df_owner['owner'][i].split('\n')):
                    position = remove_punctuation_withSpace(translate_to_english(name.split(' - ')[-2].lower()).replace('.',''))
                    if(check_position(position)):
                        parsed_name = HumanName(name.split(' - ')[0])
                        strung = remove_punctuation(parsed_name.first+' '+parsed_name.last+'\n')
                        ok = 0
                        if(df_owner.req_retry[i]<4):
                            for search in json.loads(df_owner.text_data[i])['organic_results']:
                                if(name.split(' - ')[0].replace('-',' ').lower() in search['title'].replace('-',' ').lower()):
                                    ok=1
                                    strung = strung + search['link']
                                    break
                        if(ok==0):
                            owner_linkedin, other_linkedin = find_linkedin(name,df_owner['text_data'][i])
                            #owner_linkedin =""
                            if(len(owner_linkedin)!=0):
                                strung = strung + owner_linkedin[0]
                            else:
                                strung = strung + '---'
                        name_array.append(strung)
                        rank_array.append(name.split(' - ')[-2])
                        company_array.append(name.split(' - ')[-1])
                df_owner['name'][i] = "\n".join(name_array)
                df_owner['rank'][i] = "\n".join(rank_array)
                df_owner['company_found'][i] = "\n".join(company_array)
                if(len(df_owner['name'][i])==0):
                    df_owner['name'][i] = '---'
                if(len(df_owner['rank'][i])==0):
                    df_owner['rank'][i] = '---'
                if(len(df_owner['company_found'][i])==0):
                    df_owner['company_found'][i] = '---'
                print(df_owner['name'][i])
                print(df_owner['rank'][i])
                print(df_owner['company_found'][i])
        with df_lock:
            df_csv = pd.read_csv(path+'\\mid.csv')
            df_csv['owner'][i] = df_owner['owner'][i]
            df_csv['OtherLinkedin'][i] = df_owner['OtherLinkedin'][i]
            df_csv['OwnerLinkedin'][i] = df_owner['OwnerLinkedin'][i]
            df_csv['gpt_retry'][i] = df_owner['gpt_retry'][i]
            df_csv['name'][i] = df_owner['name'][i]
            df_csv['rank'][i] = df_owner['rank'][i]
            df_csv['company_found'][i] = df_owner['company_found'][i]
            for j in range(len(df_owner)):
                if(df_owner['company'][j]==df_owner['company'][i] and j!=i):
                    df_owner['owner'][j] = df_owner['owner'][i]
                    df_owner['OtherLinkedin'][j] = df_owner['OtherLinkedin'][i]
                    df_owner['OwnerLinkedin'][j] = df_owner['OwnerLinkedin'][i]
                    df_owner['gpt_retry'][j] = -1
                    df_owner['name'][j] = df_owner['name'][i]
                    df_owner['rank'][j] = df_owner['rank'][i]
                    df_owner['company_found'][j] = df_owner['company_found'][i]
                    df_csv['owner'][j] = df_owner['owner'][i]
                    df_csv['OtherLinkedin'][j] = df_owner['OtherLinkedin'][i]
                    df_csv['OwnerLinkedin'][j] = df_owner['OwnerLinkedin'][i]
                    df_csv['gpt_retry'][j] = -1
                    df_csv['name'][j] = df_owner['name'][i]
                    df_csv['rank'][j] = df_owner['rank'][i]
                    df_csv['company_found'][j] = df_owner['company_found'][i]
            print('Company', df_owner['company'][i])        
            print(i,df_owner['owner'][i])
            print(df_owner['OwnerLinkedin'][i])
            df_csv.to_csv(path+'\\mid.csv',index=False, encoding = "utf-8-sig")
        semaphore.release()
    try:
        # use ThreadPoolExecutor instead of ProcessPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            for i in range(len(df_owner)):
                semaphore.acquire()
                executor.submit(process_row, i)
            for _ in range(3):
                semaphore.acquire()
    finally:
        # Close the executor
        executor.shutdown(wait=False)
    
    return df_owner


def main(path):
    import time
    import os
    # path = r"C:\Users\40757\Desktop\SergiuShortcut\scrappers\Google\GoogleSearchCompany"
    while True:
        time.sleep(5)
        find_owners(path)
        if os.path.exists(path+'\\done.txt'):
            time.sleep(5)
            find_owners(path)
            break
    with open(path+'\\done1.txt', 'w') as f:
        pass
    print("???????????????????????????????????????????FINISHED FINDING OWNERS???????????????????????????????????????????")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    
    main(args.path)
