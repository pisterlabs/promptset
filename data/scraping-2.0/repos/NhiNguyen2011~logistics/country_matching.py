import sys
import pandas as pd
import numpy as np
import os
import re
import openai
from dotenv import load_dotenv, find_dotenv
import langid
from datetime import datetime

_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')

abbr_dict = {
    'Albanien': 'AL',
    'Andenstaaten': 'AND',
    'Kolumbien': 'CO',
    'Ecuador': 'EC',
    'Peru': 'PE',
    'Bosnien und Herzegowina': 'BA',
    'Kanada': 'CA',
    'CARIFORUM': 'CAF',
    'Antigua und Barbuda': 'AG',
    'Barbados': 'BB',
    'Bahamas': 'BS',
    'Belize': 'BZ',
    'Dominica': 'DM',
    'Dominikanische Republik': 'DO',
    'Grenada': 'GD',
    'Guyana': 'GY',
    'Haiti': 'HT',
    'Jamaika': 'JM',
    'St. Kitts und Nevis': 'KN',
    'St. Lucia': 'LC',
    'Suriname': 'SR',
    'Trinidad und Tobago': 'TT',
    'St. Vincent': 'VC',
    'Zentralamerika': 'CAM',
    'Costa Rica': 'CR',
    'Guatemala': 'GT',
    'Honduras': 'HN',
    'Nicaragua': 'NI',
    'Panama': 'PA',
    'El Salvador': 'SV',
    'Zentralafrika oder Kamerun': 'CAS',
    'Kamerun': 'CM',
    'Schweiz': 'CH',
    'Elfenbeinküste — Ivory Coast — Costa d\'Avorio': 'CI',
    'Chile': 'CL',
    'Algerien': 'DZ',
    'Ägypten': 'EG',
    'Länder des östlichen südlichen Afrikas': 'ESA',
    'Komoren': 'KM',
    'Madagaskar': 'MG',
    'Mauritius': 'MU',
    'Seychellen': 'SC',
    'Simbabwe': 'ZW',
    'Sambia': 'ZM',
    'Färöer': 'FO',
    'Great Britain oder United Kingdom': 'GB',
    'Georgien': 'GE',
    'Ghana': 'GH',
    'Israel': 'IL',
    'Island': 'IS',
    'Jordanien': 'JO',
    'Japan': 'JP',
    'Südkorea': 'KR',
    'Libanon': 'LB',
    'Lichtenstein': 'LI',
    'Marokko': 'MA',
    'Moldawien': 'MD',
    'Montenegro': 'ME',
    'Nord-Mazedonien': 'MK',
    'Mexiko': 'MX',
    'Norwegen': 'NO',
    'Westjordanland und Gazastreifen': 'PS',
    'Serbien': 'XS oder RS',
    'Entwicklungsgemeinschaft des südlichen Afrikas': 'SADC',
    'Botsuana': 'BW',
    'Lesotho': 'LS',
    'Mosambik': 'MZ',
    'Namibia': 'NA',
    'Swasiland': 'SZ',
    'Südafrika': 'ZA',
    'Singapur': 'SG',
    'Tunesien': 'TN',
    'Ukraine': 'UA',
    'Überseeische Länder und Gebiete': 'ÜLG/OCT',
    'Aruba': 'AW',
    'Saint-Barthélemy': 'BL',
    'Bonaire': 'BQ',
    'Curaçao': 'CW',
    'Grönland': 'GL',
    'Neukaledonien': 'NC',
    'Französisch-Polynesien': 'PF',
    'St. Pierre und Miquelon': 'PM',
    'St. Martin': 'SX',
    'Französische Süd- und Antarktisgebiete': 'TF',
    'Wallis und Futuna': 'WF',
    'Vietnam': 'VN',
    'West-Pazifik-Staaten': 'WPS',
    'Fidji': 'FJ',
    'Papua-Neuguinea': 'PG',
    'Salomonen': 'SB',
    'Samoa': 'WS',
    'Ceuta': 'XC',
    'Kosovo': 'XK',
    'Melilla': 'XL',
    'Overseas Countries and Territories': 'OCT',
    'Eastern and Southern Africa': 'ESA',
    'Cariforum': 'CAF',
    'Southern African Development Community': 'SADC',
    'Western Pacific States': 'WPS',
    'European Economic Area': 'EEA',
    'Central America': 'CAM',
    'Central Africa': 'CAS',
    'Andean countries': 'AND'
}

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def required_countries_zonen(file_path):
    zonen_dict = {}
    group_dict = {}

    # Open and read the text file   
    with open(file_path, 'r', encoding="utf-8") as file:
        lines = file.readlines()     
        
    # Process the lines to create the dictionary
    for line in lines:
        parts = line.strip().split(' - ')  
        key = parts[0] 
        values = [parts[1]] 
        
        if len(parts) > 2:
            group = parts[1]
            members = parts[2:]
            group_dict[group] = members   
        
        # Check if the key already exists in the dictionary
        if key in zonen_dict:
            # If the key exists, append the new values to the existing list of values
            zonen_dict[key].extend(values)
        else:
            # If the key is new, create a new entry in the dictionary
            zonen_dict[key] = values
    
    return zonen_dict, group_dict

def required_countries(file_path):
    # criteria for only countries
    group_dict = {}
    countries = []

    # Open and read the text file   
    with open(file_path, 'r', encoding="utf-8") as file:
        lines = file.readlines()

    # Process the lines to create the dictionary
    for line in lines:
        parts = line.strip().split(' - ')  
        countries.append(parts[0])

        if len(parts) > 1:
            group = parts[0]
            members = parts[1:]
            group_dict[group] = members  
    return countries, group_dict

def get_text(input_file):
    with open(input_file, 'r') as file:
        text = file.read()
    return text

def detect_languages(text):
    # check whether document is in english or german
    all_langs = [lang for lang, _ in langid.rank(text)]
    return all_langs[:2]

def get_country_text(text):
    prompt= f"""
    You will receive a text.
    Return exactly the part of text that contains the list of countries organized into specific regional groupings or partnerships in various languages, 
    together with its notes marked with a visual delimiter or marker at the end if exist.
    Don't repeat the prompt in the result. Don't change anything from the original text.
    Text:
    ```{text}```
    """

    response = get_completion(prompt)
    return response

def process_country_text(country_text, group_dict):
    # Extract country code from document 
    tf_countries = []
    # exclude the group that applies transitional rules
    if ("transitional rules" in country_text.lower()) or ("übergangsbestimmungen" in country_text.lower()):
        prompt = f"""
        You will receive a list of countries organized into specific regional groupings or partnerships in various languages.
        Identify the country group that applies "Transitional rules". 
        The keyword "Transitional rules"("Übergangsbestimmungen" in German) may appear either at the line before or at the end of the line.
        It can also be marked with a visual delimiter or marker indicating that "Transitional rules" apply to all countries in that line.
        Find the entire line mentioning the country group and its associated countries that apply "Transitional rules" in the given text.
        Return the line as plaintext enclosed within triple backticks, excluding the line "Transitional rules (Gilt für alle Länder in dieser Zeile)"

        Text:
        ```{country_text}```
        """
        
        rm_text = get_completion(prompt)
        try:
            match = re.search(r"```(.+?)```", rm_text, re.DOTALL)  # match newline characters as well.
            if match:
                transitional_rule = match.group(1).strip()
                tf_countries = re.findall(r'\b([A-Z]{2})\b', transitional_rule)
                
                # remove the line that applies transitional rules in the text
                start_index_tr = country_text.find(transitional_rule)
                end_index_tr = start_index_tr + len(transitional_rule)
                country_text = country_text[:start_index_tr] + country_text[end_index_tr:]
            else:
                raise ValueError("Error: Countries applying transitional rules not found in the text.")

        except ValueError as e:
            print(e)

    country_text = country_text.upper()   
    pattern = re.compile(r'\b([A-Z]{2,4})\b') # set pattern to find country code

    # extract the country codes using API OpenAI
    prompt = f"""
    Given a text containing country names and specific regional groupings or partnerships in various languages, 
    transform the country names and regional groupings into corresponding abbreviations based on the provided dictionary '''{abbr_dict}'''.          
    Provide the corresponding abbreviations in a list format, not the long forms of country 
    Text:
    ```{country_text}```
    """
    result = get_completion(prompt)
    input_countries_gpt = re.findall(pattern,result)
    input_countries_re = re.findall(pattern,country_text)
    input_countries = list(set(input_countries_gpt + input_countries_re))

    # Either only the country group name present, or all belonging countries must present
    # Exception for SADC
    for key, values in group_dict.items():
        if key in input_countries: 
            if key == 'SADC':
                common_value = [value for value in values if value in input_countries]
                if (not common_value) or (common_value == ['ZA']):
                    continue
                else:
                    input_countries.remove(key)
            elif any(value in input_countries for value in values):
                input_countries.remove(key)
    
    return input_countries, tf_countries

def check_countries(countries, group_dict, input_countries, log_de, log_en):
    missing_countries = []
    for country in countries:
        match2 = re.compile(r'(\w+)\s*\/\s*(\w+)', flags=re.IGNORECASE).search(country)
        if match2:
            if (match2.group(1) not in input_countries) and (match2.group(2) not in input_countries):
                missing_countries.append(country)
            continue
        if country not in input_countries:
            missing_countries.append(country)
    for missing in missing_countries:
        if missing in group_dict.keys():
            missing_member = []
            for member in group_dict[missing]:
                if member not in input_countries:
                    missing_member.append(member)
            if not missing_member:
                missing_countries = [x for x in missing_countries if x != missing]
            elif (missing_member == ["HT"]) or (missing_member == ["ZM"]):
                missing_countries = [x for x in missing_countries if x != missing]
                text_de = f"Bei der Gruppe {missing} fehlt das Land {missing_member}, ist aber noch nicht anwendbar."
                text_en = f"In the {missing} group, the country {missing_member} is missing but not yet applicable."
                log_de.append(text_de)
                log_en.append(text_en)
            elif len(missing_member) != len(group_dict[missing]):
                text_de = f"Gruppe {missing} ist unvollständig, fehlende Länder: {missing_member}"
                text_en = f"Group {missing} is incomplete, missing countries: {missing_member}"
                if any(member == "HT" or member == "ZM" for member in missing_member):
                    text_de += f" aber {', '.join(member for member in missing_member if member == 'HT' or member == 'ZM')} noch nicht anwendbar."
                    text_en += f" but {', '.join(member for member in missing_member if member == 'HT' or member == 'ZM')} is not applicable yet."
                log_de.append(text_de)
                log_en.append(text_en)
    return missing_countries

def matching_countries(countries, group_dict, input_countries, zonen_dict, tf_countries, selected_option):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    log_de = []
    log_en =[] 

    if selected_option == "Allgemein ohne Wirtschaftszonen":
        missing_countries = check_countries(countries, group_dict, input_countries, log_de, log_en)

        if missing_countries:
            text_de = f"Fehlende Länder/Ländergruppen: {missing_countries}"
            text_en = f"Missing countries/country groups: {missing_countries}"
            if "OCT/ÜLG" in missing_countries:
                text_de += f" aber OCT/ÜLG noch nicht anwendbar."
                text_en += f" but OCT/ÜLG is not applicable yet."  
            log_de.append(text_de)
            log_en.append(text_en)

    else:
        for economic_area, zone_countries in zonen_dict.items():
            missing_countries = check_countries(zone_countries, group_dict, input_countries, log_de, log_en)
            if missing_countries:
                if selected_option == "Avient Color":
                    if economic_area == "0":
                        log_de.append(f"Fehlende Länder/Ländergruppen: {missing_countries} aber noch nicht anwendbar")
                        log_en.append(f"Missing countries/country groups: {missing_countries} but not yet applicable") 
                    else:
                        log_de.append(f"Die Wirtschaftsgebiet {economic_area} ist unvollständig. Fehlende Länder/Ländergruppen: {missing_countries}")
                        log_en.append(f"The economic zone {economic_area} is incomplete. Missing countries/country groups: {missing_countries}")
                
                elif selected_option == "Clariant Gruppe + Heubach":
                    text_de_zone = f"Die Wirtschaftsgebiet {economic_area} ist unvollständig. Fehlende Länder/Ländergruppen: {missing_countries}"
                    text_en_zone = f"The economic zone {economic_area} is incomplete. Missing countries/country groups: {missing_countries}"
                    if "ÜLG/OCT" in missing_countries:
                        text_de_zone = text_de_zone + f" aber ÜLG/OCT noch nicht anwendbar."
                        text_en_zone = text_en_zone + f" but ÜLG/OCT is not applicable yet."            
                    log_de.append(text_de_zone)
                    log_en.append(text_en_zone)

                elif selected_option == "Avient Luxembourg":
                    if economic_area == "Keine Zuordnung":
                        log_de.append(f"Fehlende Länder/Ländergruppen: {missing_countries} aber noch nicht anwendbar")
                        log_en.append(f"Missing countries/country groups: {missing_countries} but not yet applicable") 
                    else:
                        log_de.append(f"Die Wirtschaftsgebiet {economic_area} ist unvollständig. Fehlende Länder/Ländergruppen: {missing_countries}")
                        log_en.append(f"The economic zone {economic_area} is incomplete. Missing countries/country groups: {missing_countries}")                    

                    
                #Check countries apllying transitional rules
                ignored_tf_countries = [tf_country for tf_country in tf_countries if tf_country in missing_countries]
                if ignored_tf_countries:
                    log_de.append(f"{ignored_tf_countries} sind/ist zwar genannt, jedoch im Rahme von Transitional Rules, welches bei uns noch nicht angewendet werden." )
                    log_en.append(f"{ignored_tf_countries} are/is indeed mentioned but within Transitional Rules, which has not yet been applied.")
                    

    if not log_de:
        log_de.append("Alles in Ordnung")
        log_en.append("Everything is in order")

    for item in log_de:
        print(item)
    print(f"Ausführungsdatum und -zeit: {formatted_datetime}")
    print( )

    for item in log_en:
        print(item)
    print(f"Execution date and time: {formatted_datetime}")      
                
template_keywords_en = [
    "Long-term supplier's declaration for products having preferential origin status",
    "Declaration",
    "I, the undersigned, declare that the goods described",
    "which are regularly supplied to",
    "originate in",
    "satisfy the rules of origin governing preferential trade with",
    "I declare that",
    "This declaration is valid for all shipments of these products dispatched from",
    "I undertake to inform",
    "immediately",
    "if this declaration is no longer valid",
    "I undertake to make available to the customs authorities any further supporting documents they require"
]

template_keywords_de = [
    "Lieferantenerklärung für Waren mit Präferenzursprungseigenschaft",
    "Erklärung",
    "Der Unterzeichner erklärt, dass die in diesem Dokument aufgeführten",
    "Waren Ursprungserzeugnisse",
    "sind und den Ursprungsregeln für den Präferenzverkehr",
    "entsprechen"
    "Er erklärt Folgendes",
    "Diese Erklärung gilt für alle Sendungen dieser Waren im Zeitraum",
    "Der Unterzeichner verpflichtet sich",
    "umgehend zu unterrichten",
    "wenn diese Erklärung ihre Geltung verliert",
    "Er verpflichtet sich, den Zollbehörden alle von ihnen zusätzlich verlangten Belege zur Verfügung zu stellen"
]


def check_keywords_in_document(document_text, template_keywords,template_keywords_de):
    missing = []
    
    if "declaration" in document_text.lower():
        document_text_st = document_text.lower().replace("long term","long-term").replace("\n"," ").replace("  "," ")
        for keyword in template_keywords:
            if keyword.lower() not in document_text_st:
                missing.append(keyword)
                print(f'The document does not match the template at: "{keyword}"')
        
        words = document_text.lower().split()
        word_count=words.count("cumulation")

        match = re.search(r'\bx\b',document_text, flags=re.IGNORECASE)

        if word_count > 1:
            if match:
                # Get the position of the match
                x_index = match.start()
                for i in range(x_index+1,len(document_text)):
                    if document_text[i].isalpha():
                        next_alphabet = document_text[i]
                        break

                if next_alphabet.upper() == 'C':
                    print("'No cumulation applied' is not selected")
        elif "no cumulation applied" not in document_text.lower():
                print("The document does not match the template at: 'No cumulation applied'")
                
        if not missing:
            print(f"The document matches the template.") 

    elif "Erklärung" in document_text:
        document_text_st = document_text.replace("\n"," ").replace("  "," ")
        for keyword in template_keywords:
            if keyword not in document_text_st:
                missing.append(keyword)
                print(f'"Das Dokument entspricht nicht der Vorlage an der Stelle: "{keyword}"')
        
        words = document_text.lower().split()
        word_count=words.count("Kumulierung")

        match = re.search(r'\bX\b',document_text, flags=re.IGNORECASE)

        if word_count > 1:
            if match:
                # Get the position of the match
                x_index = match.start()
                for i in range(x_index+1,len(document_text)):
                    if document_text[i].isalpha():
                        next_alphabet = document_text[i]
                        break

                if next_alphabet.upper() == 'KU':
                    print("'Keine Kumulierung angewendet' ist nicht ausgewählt")
        elif "Keine Kumulierung angewendet" not in document_text.lower():
                print("Das Dokument entspricht nicht der Vorlage an der Stelle: 'Keine Kumulierung angewendet'")
                
        if not missing:
            print(f"Das Dokument entspricht der Vorlage.") 
    
    else:
        raise ValueError("Unsupported language detected/ Sprache unerkannt")

def main():
    text = get_text(input_file)
    country_text = get_country_text(text)
    print(country_text)    

    if selected_option == "Allgemein ohne Wirtschaftszonen":
        countries, group_dict = required_countries(file_path)
        zonen_dict = {}
    else:
        countries = []
        zonen_dict, group_dict = required_countries_zonen(file_path)

    input_countries, tf_countries = process_country_text(country_text, group_dict)
    
    matching_countries(countries, group_dict, input_countries, zonen_dict, tf_countries, selected_option)
    check_keywords_in_document(text, template_keywords_en,template_keywords_de)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_script.py input_file")
        sys.exit(1)

    input_file = sys.argv[1]
    selected_option = sys.argv[2]

    if selected_option == "Allgemein ohne Wirtschaftszonen":
        file_path = 'Ländervorgaben_Allgemein.txt' 
    elif selected_option == "Clariant Gruppe + Heubach":
        file_path = 'Ländervorgaben_AvientClariantGruppeHeubach.txt' 
    elif selected_option == "Avient Color":
        file_path = 'Ländervorgaben_AvientColor.txt' 
    elif selected_option == "Avient Luxembourg":
        file_path = 'Ländervorgaben_AvientLuxembourg.txt'     
    else:
        print(f"Unknown option: {selected_option}")
        sys.exit(1)

    main()

            
