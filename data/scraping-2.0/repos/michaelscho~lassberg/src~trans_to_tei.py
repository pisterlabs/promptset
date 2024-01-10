# imports

import config

import re
import time
import os
from datetime import datetime

import requests
from zipfile import ZipFile
import lxml.etree as LET
import pandas as pd

import openai
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.nn import Classifier
from flair.data import Label

openai.api_key = config.openai_key



""" This script exports page-xml files from Transkribus and processes them to TEI-XML files via NER and gpt4 processing.
"""

class GetDataFromXml:
    """ This class loads the xml files with the register data and creates a list of persons and places for further processing.

    """
    def __init__(self):
        # load person register into list
        self.root_persons = self.load_and_get_root('lassberg-persons.xml')
        self.list_of_persons = self.create_list_from_xml(self.root_persons, 'person')

        # load place register into list
        self.root_places = self.load_and_get_root('lassberg-places.xml')
        self.list_of_places = self.create_list_from_xml(self.root_places, 'place')

        # create gpt lookup strings with less tokens to reduce costs
        self.persons_gpt_lookup = self.create_gpt_lookup(self.list_of_persons)
        self.places_gpt_lookup = self.create_gpt_lookup(self.list_of_places)

    def create_gpt_lookup(self, list):
        # create new list from register files with only number of id and name to reduce tokens used
        list_for_gpt = []
        for item in list:
            list_for_gpt.append([item[0].replace('lassberg-place-','').replace('lassberg-correspondent-',''), item[2]])
        # create string representation from list
        string_for_gpt = ''
        for item in list_for_gpt:
            string_for_gpt += f'{item[0]};{item[1]}|'
        
        return string_for_gpt
        
    def load_and_get_root(self, file):
        """ This helper function loads the xml file and returns the root element."""

        xml_file = os.path.join(os.getcwd(), '..', 'data', 'register', file)
        tree = LET.parse(xml_file)
        root = tree.getroot()

        return root

    def create_list_from_xml(self, root, element):
        """ This helper function creates a list from a xml register file."""

        list_of_items = []
        for item in root.findall('.//{*}' + element):
            if element == 'person':
                list_of_items.append([item.get('{http://www.w3.org/XML/1998/namespace}id'), item.get('ref'), re.sub('\s+',' ', item.find('.//{*}persName').text.replace('\n',''))])
            if element == 'place':
                list_of_items.append([item.get('{http://www.w3.org/XML/1998/namespace}id'), item.find('.//{*}placeName').get('ref'), re.sub('\s+',' ', item.find('.//{*}placeName').text.replace('\n',' '))])
        
        return list_of_items


class GetDataFromCsv:
    """ This class loads the csv file with the register containing metadata for each letter and creates a pandas dataframe for further processing.
    """
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.letter_data = self.read_register_csv()

    def read_register_csv(self):
        """ This function reads the csv file with the register data and returns a letter as a pandas dataframe.
        """
        # read in register file
        register = pd.read_csv(os.path.join(os.getcwd(), '..', 'data', 'register', 'register.csv'), sep=';', encoding='utf8')
        # find row with doc_id
        row = register.loc[register['ID'] == self.doc_id]
        
        return row


class CreateXML:
    """ This class creates the xml file for each letter, inserts processed data and saves it to the folder ../data/letters/lassberg-letter-id.xml.
    """
    def __init__(self, doc_id, letter_data, original_text, normalized_text, translated_text, summary_text, list_of_persons, list_of_places):
        self.list_of_persons = list_of_persons
        self.list_of_places = list_of_places
        self.original_text = original_text
        
        self.normalized_text = normalized_text
        self.translated_text = translated_text  
        self.summary_text = summary_text
        
        self.doc_id = doc_id
        self.letter_data = letter_data
        self.xml_template = self.create_letter()
        self.encoded_letter = self.replace_placeholder_in_template()
        self.create_letter()
        self.save_letter()

    def replace_placeholder_in_template(self):
        """ This function replaces the placeholders in the xml template with the data from the register and the processed text.
        """
        
        self.xml_template = self.xml_template.replace('{lassberg-letter-XML_ID}',self.doc_id)
        self.xml_template = self.xml_template.replace('{SENT_BY}',self.letter_data['SENT_FROM_NAME'].values[0])
        self.xml_template = self.xml_template.replace('{SENT_TO}',self.letter_data['RECIVED_BY_NAME'].values[0])
        self.xml_template = self.xml_template.replace('{SENT_DATE}', datetime.strptime(self.letter_data['Datum'].values[0], '%Y-%m-%d').strftime('%d.%m.%Y'))

        self.xml_template = self.xml_template.replace('{REPOSITORY_PLACE}',self.letter_data['Aufbewahrungsort'].values[0])
        self.xml_template = self.xml_template.replace('{REPOSITORY_INSTITUTION}',self.letter_data['Aufbewahrungsinstitution'].values[0])
        self.xml_template = self.xml_template.replace('{REPOSITORY_SIGNATURE}',self.letter_data['Signatur'].values[0].strip())
        self.xml_template = self.xml_template.replace('{REGISTER_HARRIS}',str(self.letter_data['Nummer_Harris'].values[0]))
        self.xml_template = self.xml_template.replace('{REGISTER_LASSBERG}',self.letter_data['Journalnummer'].values[0])

        self.xml_template = self.xml_template.replace('{PRINTED_IN}',self.letter_data['published_in'].values[0])
        self.xml_template = self.xml_template.replace('{PRINTED_IN_URL}',self.letter_data['published_in_url'].values[0])
        self.xml_template = self.xml_template.replace('{XML_ID}',self.doc_id)
        self.xml_template = self.xml_template.replace('{SENT_DATE_ISO}',self.letter_data['Datum'].values[0])

        try:
            self.xml_template = self.xml_template.replace('{ORIGINAL_TEXT}', self.original_text)
        except: 
            pass

        try:
            self.xml_template = self.xml_template.replace('{NORMALIZED_TEXT}', self.normalized_text)
        except: 
            pass

        try:
            self.xml_template = self.xml_template.replace('{TRANSLATED_TEXT}', self.translated_text)
        except: 
            pass

        try:
            self.xml_template = self.xml_template.replace('{SUMMARY_TEXT}', self.summary_text)
        except: 
            pass

        self.xml_template = self.xml_template.replace('{today}',datetime.today().strftime('%Y-%m-%d'))
        
        # look up list of persons based on correspondent partners id
        matching_correspondent_to = [sublist for sublist in self.list_of_persons if self.letter_data['RECIVED_BY_ID'].values[0] in sublist]
        matching_correspondent_from = [sublist for sublist in self.list_of_persons if self.letter_data['SENT_FROM_ID'].values[0] in sublist]

        if self.letter_data['SENT_FROM_NAME'].values[0] == 'Joseph von Laßberg':
            self.xml_template = self.xml_template.replace('{PERS_TO_NUMBER}\" ref=\"{GND}\"', f'{self.letter_data["RECIVED_BY_ID"].values[0]}\" ref=\"{matching_correspondent_to[0][1]}\"')
            self.xml_template = self.xml_template.replace('{PERS_FROM_NUMBER}\" ref=\"{GND}\"', f'lassberg-correspondent-0373\" ref=\"https://d-nb.info/gnd/118778862\"')
            self.xml_template = self.xml_template.replace('<placeName key="../register/lassberg-places.xml#lassberg-place-{PLACE_FROM_NUMBER}" ref="{PLACE_FROM_METADATA}">{PLACE_SENT_FROM}</placeName>', f'<placeName key=\"../register/lassberg-places.xml#{ self.letter_data["Absendeort_id"].values[0] }\">{ self.letter_data["Absendeort"].values[0] }</placeName>')
            self.xml_template = self.xml_template.replace('<placeName key="../register/lassberg-places.xml#lassberg-place-{PLACE_TO_NUMBER}" ref="{PLACE_TO_METADATA}">{PLACE_SENT_TO}</placeName>', '')
        
        else:
            self.xml_template = self.xml_template.replace('{PERS_FROM_NUMBER}\" ref=\"{GND}\"', f'{self.letter_data["SENT_FROM_ID"].values[0]}\" ref=\"{matching_correspondent_from[0][1]}\"')
            self.xml_template = self.xml_template.replace('{PERS_TO_NUMBER}\" ref=\"{GND}\"', f'lassberg-correspondent-0373\" ref=\"https://d-nb.info/gnd/118778862\"')
            self.xml_template = self.xml_template.replace('<placeName key="../register/lassberg-places.xml#lassberg-place-{PLACE_FROM_NUMBER}" ref="{PLACE_FROM_METADATA}">{PLACE_SENT_FROM}</placeName>', f'<placeName key=\"../register/lassberg-places.xml#{ self.letter_data["Absendeort_id"].values[0] }\">{ self.letter_data["Absendeort"].values[0] }</placeName>')
            self.xml_template = self.xml_template.replace('<placeName key="../register/lassberg-places.xml#lassberg-place-{PLACE_TO_NUMBER}" ref="{PLACE_TO_METADATA}">{PLACE_SENT_TO}</placeName>', f'')

        encoded_letter = self.xml_template

        return encoded_letter

    def create_letter(self):
        # read in xml template as string
        with open(os.path.join(os.getcwd(), '..', 'data','letter_template.xml'), 'r', encoding='utf8') as f:
            xml_file = f.read()

        return xml_file

    def save_letter(self):
        # save xml file to folder ../data/letters
        with open(os.path.join(os.getcwd(), '..', 'data', 'letters', f'{self.doc_id}.xml'), 'w', encoding='utf8') as f:
            f.write(self.encoded_letter)


class ProcessPageXML:
    """ This class processes the page-xml files from Transkribus, processes text using gpt3.5 or gpt4 
        via openai and creates a TEI representation of the letter.
    """

    def __init__(self, doc_title, gpt_version, list_of_persons, list_of_places, persons_gpt_lookup, places_gpt_lookup, log):
        self.gpt_version = gpt_version
        self.doc_title = doc_title
        self.persons_gpt_lookup = persons_gpt_lookup
        self.places_gpt_lookup = places_gpt_lookup
        self.letter_text = self.create_tei_from_pagexml()
        
        #self.letter_normalization = self.create_normalisation()
        #self.letter_translation = self.create_translation()
        #self.letter_summary = self.create_summary()
        
        self.tags = []

        self.list_of_persons = list_of_persons
        self.list_of_places = list_of_places

        self.max_id_persons = max([int(item[0].replace('lassberg-correspondent-','')) for item in self.list_of_persons])
        self.max_id_places = max([int(item[0].replace('lassberg-place-','')) for item in self.list_of_places])

        self.list_of_entities = self.ner_extraction(log)
        
        #self.ner_processing_computed()
        self.ner_processing_llm(log)

        self.ner_replacement(log)
        self.create_tei_file()

    def create_tei_from_pagexml(self):
        """
        Creates TEI representation from the extracted text in the PAGE XML files.

        This function processes each PAGE XML file and extracts the text from the columns and headers.
        It constructs the TEI representation by combining the extracted text and applying certain
        replacements and transformations. The TEI representation is stored in the 'bdd_tei_text' attribute of the object.
        """

        letter_text = ""
        # open folder with pagexml-files
        path_to_folder = os.path.join(os.getcwd(),'..','data','pagexml',f'{self.doc_title}')
        path_to_pagexml_files = [os.path.join(path_to_folder,f) for f in os.listdir(path_to_folder)]

        # open each pagexml-file for exporting text to tei
        n = 1
        lb_break = False
        for filename in path_to_pagexml_files:
            tree = LET.parse(filename)
            root = tree.getroot()
            page_text = ""

            # creates page beginning for each xml-pagefile using data taken from config file
            page_break = f'<pb n="{n}" corresp="../pagexml/{self.doc_title}/{self.doc_title}-{n}.xml"/>'
            page_text = page_text + page_break    
            # iterate over all text regions
            for text_region in root.findall('.//{*}TextRegion'):
                # iterate over all text lines
                e = 1
                for text_line in text_region.findall('.//{*}TextLine'):
                        
                    # ad <lb> element and text
                    break_equals_no = "break=\"no\" "
                    line_content = f"<lb {break_equals_no if lb_break == True else ''}xml:id=\"{self.doc_title}-{n}-{e}\" n=\"{e}\" corresp=\"{text_line.get('id')}\"/>{text_line.findall('.//{*}Unicode')[0].text}\n"
                    e = e + 1
                    page_text = page_text + line_content
                    if '¬' in text_line.findall('.//{*}Unicode')[0].text:
                        lb_break = True
                        page_text = page_text.replace('¬','')
                    else:
                        lb_break = False

            n = n + 1

            letter_text = letter_text + page_text
        
        return letter_text

    def create_normalisation(self, log):
        # prompt openai api for gpt4 to get summary
        content = "Der folgende Brief ist aus dem 19. Jahrhundert und verwendet eine altertümliche deutsche Ortographie. Bitte gebe den Brief nach moderner Rechtschreiberegel wieder: " + self.letter_text
        messages = [{"role": "user", "content": content}]
        completion = openai.ChatCompletion.create(model=self.gpt_version,messages=messages)
        letter_normalization = completion.choices[0].message.content
        log.log(f"\nNormalised letter (GPT4): \n{letter_normalization}")

        return letter_normalization

    def create_summary(self, log):
        content = "Fasse den folgenden Brief in 100 Worten zusammen: " + self.letter_normalization
        messages = [{"role": "user", "content": content}]
        completion = openai.ChatCompletion.create(model=self.gpt_version,messages=messages)
        letter_summary = completion.choices[0].message.content
        log.log(f"\nSummary (GPT4): \n{letter_summary}")

        return letter_summary

    def create_translation(self, log):
        content = "Übersetze den folgenden Brief in Englisch: " + self.letter_normalization
        messages = [{"role": "user", "content": content}]
        completion = openai.ChatCompletion.create(model=self.gpt_version,messages=messages)
        letter_translation = completion.choices[0].message.content
        log.log(f"\Translation (GPT4): \n{letter_translation}")

        return letter_translation

    def extract_tags(self, text):
        """Extracts tags and their positions from the text."""
        
        for match in re.finditer(r"<[^>]+>", text):
            self.tags.append((match.start(), match.end(), match.group()))
        
    def remove_tags(self, text):
        """Removes tags from the text."""
        for position_start, position_end, tag in reversed(self.tags):  # Start from the end to not mess up positions
            text = text[:position_start] + text[position_end:]
        return text

    def reinsert_tags(self, text):
        """Reinserts tags into their original positions in the text."""
        for position_start, position_end, tag in self.tags:
            text = text[:position_start] + tag + text[position_end:]
        return text

    def ner_extraction(self, log):
        print(f"\nStarting NER extraction for {self.doc_title}")
        log.log(f"\nStarting NER extraction for {self.doc_title}")

        self.letter_text = re.sub('\n<lb break','<lb break', self.letter_text)
        self.letter_text = re.sub('\n<lb x',' <lb x', self.letter_text)
        tags = self.extract_tags(self.letter_text)
        self.letter_text = self.remove_tags(self.letter_text)

        # NER using Flair
        # load model
        tagger = Classifier.load('de-ner-large')
        tagger.to('cpu')
    
        # make example sentence in any of the four languages
        sentence = Sentence(self.letter_text)

        # predict NER tags
        tagger.predict(sentence)

        list_of_entities = []

        # print predicted NER spans
        for entity in sentence.get_spans('ner'):
            tag: Label = entity.labels[0]
            #print(f'{entity.text} [{tag.value}] ({tag.score:.4f})')
            list_of_entities.append([entity.text, tag.value])
    
        #print(list_of_entities)
        return list_of_entities

    # TODO: implement function

    def ner_processing_computed(self):
        # iterate through self.list_of_entities and check
        for ner_item in self.list_of_entities:
            # if entity is in list_of_persons -> create reference
            if ner_item[1] == 'PER':
                ref = self.create_reference(ner_item)
                if ref is not None:
                    # here i need to ask if ref not null add to xml
                    ner_item.append(ref)

            # if entity is in list_of_places -> create reference
            elif ner_item[1] == 'LOC':
                ref = self.create_reference(ner_item)
                if ref is not None:
                    ner_item.append(ref)

        print(self.list_of_entities)

    def create_reference(self, entity):
        """ This function creates a reference to the register for each entity if it can be matched."""
        
        if entity[1] == 'PER':
            normalized_name = re.sub(r'\b\w+\.','',entity[0])
            normalized_name = re.sub('\s+',' ', normalized_name)
            normalized_name = ' '.join(normalized_name.split(' '))
            normalized_names = normalized_name.split(' ')
            if len(normalized_names) > 1:
                normalized_name = normalized_names[-1]

            if len(normalized_name) < 3:
                return None, None
        
            elif len(normalized_name) == 3 and (normalized_name == 'von' or normalized_name == 'vom'):
                return None, None
        
            elif normalized_name.isspace():
                return None, None
        
            elif normalized_name == '\n':
                return None, None

            print(normalized_name)

            # check if normalized name is substring of any item[2] in self.list_of_persons and return sublist
            is_present = [sublist for sublist in self.list_of_persons if normalized_name in sublist[2]]
            # if is_present exists -> create reference
            if is_present:
                person_id = is_present[0][0]
                person_name = is_present[0][2] 
                ref = f'../register/lassberg-persons.xml#{person_id}'
                print(person_name)
                return ref
    
        elif entity[1] == 'LOC':
            # check if normalized name is substring of any item[2] in self.list_of_places and return sublist
            is_present = [sublist for sublist in self.list_of_places if entity[0] in sublist[2]]
            # if is_present exists -> create reference
            if is_present:
                place_id = is_present[0][0]
                place_name = is_present[0][2] 
                ref = f'../register/lassberg-places.xml#{place_id}'
                print(place_name)
                return ref
        
        return None, None


    def ner_processing_llm(self, log):
        """ This function creates a reference to the register for each entity if it can be matched by gpt4 api.

            Requires openai api-key and may result in costs. If matches are found, they are added to the list_of_entities as a third element.
            That element either contains the id of the register entry or 'none' if no match was found.        
        """

        look_up_string_person = ''
        look_up_string_places = ''
        for item in self.list_of_entities:
            if item[1] == "PER":
                look_up_string_person = look_up_string_person + item[0] + '|'
            elif item[1] == "LOC":
                look_up_string_places = look_up_string_places + item[0] + '|'

        """
        Example letter 1015:
        
        Persons:
        look_up_string_person = "Inen|Rosenbäcker|Inen|H. v. Soumard|Holbein|Erik Holbein|Albr. Hegner|Berthold|Erchanger|Salomon III|Kirchhofer|Neugarts|Honerlage|Jos. von Laßberg|"
        GPT 3.5 returns: "none|(none)|none|none|none|none|none|none|none|0442|(none)|none|none|none|none|none|none|0440|(none)|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|none|0459|(none)|none|none|0462|(none)|none"
        GPT 4 returns: "Inen(none)|Rosenbäcker(none)|Inen(none)|H. v. Soumard(none)|Holbein(0354)|Erik Holbein(none)|Albr. Hegner(none)|Berthold(0385)|Erchanger(none)|Salomon III(0401)|Kirchhofer(0307)|Neugarts(0379)|Honerlage(none)|Jos. von Laßberg(0373)"

        Places:

        GPT 3.5 returns: "Constanz (0082); Worblingen (none); Selhauses (none); Ravenburg (0123); Worblingen (none); Schiener Berg (none); Schrotzburg (none); Diepoldsburg (none); Schinen (none); Stein (0144); teutschlande (none); Eppishausen (0043); Constantz (none)."
        GPT 4 returns: "Constanz(0082)|Worblingen(none)|Selhauses(none)|Ravenburg(0123)|Worblingen(none)|Schiener Berg(none)|Schrotzburg(none)|Diepoldsburg(none)|Schinen(none)|Stein(0144)|teutschlande(0187)|Eppishausen(0043)|Constantz(0082)"
        
        price for gpt 3.5: ~ 0.02 $
        price for gpt 4: ~ 0.25 $
        

        """            

        prompt_person_persons = f"""The following is a list of names of persons mentioned in a letter from the 19th century. 
                            It could refer to medieval figures as well as contemporary persons. Each person is separated by '|'. 
                            Check for each person if it might be referenced in the following register and return the list I send you with the corresponding id in '()' after each name.
                            If you cannot find a corrsponding match in the register, just return 'none' instead of the id. 
                            It is important that the list keeps being seperated by '|'. This is the list of names to check: {look_up_string_person}. This is the register in the format 'id-1; name1|id-2; name2|': {self.persons_gpt_lookup}"""

        messages = [{"role": "user", "content": prompt_person_persons}]
        
        """ Decomment for production """
        #completion = openai.ChatCompletion.create(model='gpt-4',messages=messages)
        #returned_list_persons_gpt4 = completion.choices[0].message.content

        returned_list_persons_gpt4 = "Inen(none)|Rosenbäcker(none)|Inen(none)|H. v. Soumard(none)|Holbein(0354)|Erik Holbein(none)|Albr. Hegner(none)|Berthold(0385)|Erchanger(none)|Salomon III(0401)|Kirchhofer(0307)|Neugarts(0379)|Honerlage(none)|Jos. von Laßberg(0373)"
        log.log(f"\n String returned from openai: {returned_list_persons_gpt4}")

        prompt_person_places = f"""The following is a list of placenames mentioned in a letter from the 19th century. Each place is separated by '|'. 
                            Check for each place if it might be referenced in the following register and return the list I send you with the corresponding id in '()' after each placename.
                            If you cannot find a corrsponding match in the register, just return 'none' instead of the id. 
                            It is important that the list keeps being seperated by '|'. This is the list of placenames to check: {look_up_string_places}. This is the register in the format 'id-1; placename1|id-2; placename2|': {self.places_gpt_lookup}"""

        messages = [{"role": "user", "content": prompt_person_places}]
        
        """ Decomment for production """
        #completion = openai.ChatCompletion.create(model='gpt-4',messages=messages)
        #returned_list_places_gpt4 = completion.choices[0].message.content

        returned_list_places_gpt4 = "Constanz(0082)|Worblingen(none)|Selhauses(none)|Ravensburg(0123)|Worblingen(none)|Schiener Berg(none)|Schrotzburg(none)|Diepoldsburg(none)|Schinen(none)|Stein(0144)|teutschlande(0187)|Eppishausen(0043)|Constantz(0082)"
        log.log(f"\n String returned from openai: {returned_list_places_gpt4}")
        
        # split returned lists into list of persons and list of places
        returned_list_persons_gpt4 = returned_list_persons_gpt4.split('|')
        returned_list_places_gpt4 = returned_list_places_gpt4.split('|')

        # deal with persons
        # split sublist into id and name
        for i, item in enumerate(returned_list_persons_gpt4):
            returned_list_persons_gpt4[i] = item.split('(')
            returned_list_persons_gpt4[i][1] = returned_list_persons_gpt4[i][1].replace(')','')

        for i, item in enumerate(returned_list_places_gpt4):
            returned_list_places_gpt4[i] = item.split('(')
            returned_list_places_gpt4[i][1] = returned_list_places_gpt4[i][1].replace(')','')

        # check if query returned correct number of persons and places
        if len(returned_list_persons_gpt4) != len(look_up_string_person.split('|')[:-1]):
            print('Error: Number of persons does not match')
            log.log('Error: Number of persons does not match')
        else:
            # iterate through self.list_of_entities and check if person is in returned_list_persons_gpt4
            for ner_item in self.list_of_entities:
                if ner_item[1] == "PER":
                    for item in returned_list_persons_gpt4:
                        if ner_item[0] == item[0]:
                            ner_item.append(item[1])
                            break

        # Deal with places

        # check if query returned correct number of persons and places
        if len(returned_list_places_gpt4) != len(look_up_string_places.split('|')[:-1]):
            print('Error: Number of persons does not match')
            log.log('Error: Number of persons does not match')
            
        else:
            # iterate through self.list_of_entities and check if person is in returned_list_persons_gpt4
            for ner_item in self.list_of_entities:
                if ner_item[1] == "LOC":
                    for item in returned_list_places_gpt4:
                        if ner_item[0] == item[0]:
                            ner_item.append(item[1])
                            break
                elif ner_item[1] == "PER":    
                    pass
                else:
                    ner_item.append('none')

        # check if any item in self.list_of_entities has no third element and append 'none' to it
        for ner_item in self.list_of_entities:
            if len(ner_item) < 3:
                ner_item.append('none')

    def ner_replacement(self, log):
        insertions = []  # Store list of position_start, position_end, string_to_insert)

        # Process tags
        for position_start, position_end, tag in self.tags:
            insertions.append([position_start, position_end, tag])

        # Process named entities
        print(self.list_of_entities)

        checked_places = []

        for entity, entity_type, entity_id in self.list_of_entities:
            entity = entity[:2] + '*' + entity[2:]
            for occurence in self.find_all_occurrences(self.letter_text, entity.replace('*','')):
                if entity_id == 'none':
                    key_attribute = ''
                    if entity_type == 'LOC':
                        # check is entity has already been searched for:
                        if any(entity == checked_place[0] for checked_place in checked_places):
                            # if yes, get id from checked_places
                            checked_place = [checked_place for checked_place in checked_places if checked_place[0] == entity][0]
                            if checked_place[1] == None:
                                key_attribute = ''
                            else:
                                key_attribute = f" key=\"../register/lassberg-places.xml#{checked_place[1]}\""

                        else:
                            register_entry, xml_id = self.query_wikidata(entity.replace('*',''), log)
                            checked_places.append([entity, xml_id])
                            if xml_id == None:
                                key_attribute = ''
                            else:
                                key_attribute = f" key=\"../register/lassberg-places.xml#{xml_id}\""

                else:
                    key_attribute = f" key=\"{'../register/lassberg-persons.xml#lassberg-correspondent-' if entity_type == 'PER' else '../register/lassberg-places.xml#lassberg-place-'}{entity_id}\""
                xml = f'<rs type="{entity_type}"{key_attribute}>{entity}</rs> '
                insertions.append([occurence, occurence + len(entity), xml])

        # Sort insertions by position
        insertions = [list(tup) for tup in set([tuple(x) for x in insertions])]        
        insertions.sort(key=lambda x: x[0])

        # loop through list
        for i, value in enumerate(insertions):
        # if element lb -> get index of all rs and add len(lb) to position
            if value[2].startswith("<lb") or value[2].startswith("<pb"):
                for element in insertions[i+1:]:
                    if element[2].startswith("<rs"):
                        element[0] += len(value[2])
                        element[1] += len(value[2])
            insertions.sort(key=lambda x: x[0])
                
        # if element rs -> add len(rs) to all positions except this one
            if value[2].startswith("<rs"):
                if value[0] == insertions[i+1][0]:
                    value[0] += len(insertions[i+1][2])
                    value[1] += len(insertions[i+1][2])
                    for element in insertions[i+2:]:
                        if element[2].startswith("<rs"):
                            element[0] += len(insertions[i+1][2])
                            element[1] += len(insertions[i+1][2])
                else:
                    for element in insertions[i+1:]:
                        element[0] += len(re.sub('>.*?<','><',value[2]))
                        element[1] += len(re.sub('>.*?<','><',value[2]))
                        #pass
            insertions.sort(key=lambda x: x[0])        
        
        # Insert tags into text
        for position_start, position_end, string_to_insert in insertions:
            if string_to_insert.startswith("<lb") or string_to_insert.startswith("<pb"):
                self.letter_text = self.letter_text[:position_start] + string_to_insert + self.letter_text[position_start:]
            else:
                self.letter_text = self.letter_text[:position_start] + string_to_insert + self.letter_text[position_end:]

        # normalize ner entries
        self.letter_text = self.letter_text.replace('MISC', 'misc')
        self.letter_text = self.letter_text.replace('PER','person')
        self.letter_text = self.letter_text.replace('LOC','place')
        self.letter_text = self.letter_text.replace('ORG','organisation')
        self.letter_text = self.letter_text.replace('*','')
        self.letter_text = self.letter_text.replace(' .','.')
        self.letter_text = self.letter_text.replace(' ,',',')
        self.letter_text = self.letter_text.replace(' ;',';')
        self.letter_text = self.letter_text.replace(' !','!')
        self.letter_text = self.letter_text.replace(' ',' ')


        log.log("\n" + self.letter_text)

        return self.letter_text
    

    def find_all_occurrences(self, text, substring):
        """Find all occurrences of a substring in a string."""
        start = 0
        while start < len(text):
            start = text.find(substring, start)
            if start == -1:
                break
            yield start
            start += len(substring)



    def query_wikidata(self, search_term, log):
        # Encode the search term
        encoded_search_term = requests.utils.quote(search_term)

        #print(encoded_search_term)

        # Wikidata API URL for searching entities
        api_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=de&search={encoded_search_term}&uselang=de&limit=10"

        #print(api_url)

        response = requests.get(api_url)
        search_results = response.json().get('search', [])

        #print(search_results)
        if len(search_results) == 0:
            print(f"\n-> No results found on wikidata for '{search_term}'\n")
            return None, None

        # Display the search results for user selection
        print("\nQuery from Wikidata: \n")
        for index, result in enumerate(search_results):
            label = result.get('label', 'No label')
            description = result.get('description', 'No description')
            print(f"{index+1}: {label} - {description}")
        
        # get places from self.list_of_places that start with same letters as search term
        #print(self.list_of_places)
        register_places_to_check = [place[2] for place in self.list_of_places if place[2].startswith(search_term[:2])]

        # Ask user to select a result
        selected_index = int(input(f"\n{', '.join(register_places_to_check[:20])} already in register.\nSelect place (0 for exit): \n\n"))
        if selected_index == 0:
            return None, None  # Escape mechanism
        selected_index = selected_index - 1

        
        selected_item = search_results[selected_index]
        wiki_id = selected_item.get('id')

        # Fetch coordinates
        coordinates = self.fetch_coordinates(wiki_id)

        # get wikidata link
        wikidata_link = f"https://www.wikidata.org/wiki/{wiki_id}"

        # Create XML snippet
        root = LET.Element("{http://www.tei-c.org/ns/1.0}place")

        xml_id = f"lassberg-place-{str(self.max_id_places + 1).zfill(4)}"
        self.max_id_places += 1

        root.set("{http://www.w3.org/XML/1998/namespace}id", xml_id)
        
        place_name_element = LET.SubElement(root, "{http://www.tei-c.org/ns/1.0}placeName")
        place_name_element.text = selected_item.get('label')
        place_name_element.set("ref", wikidata_link)
        
        desc_element = LET.SubElement(root, "{http://www.tei-c.org/ns/1.0}desc", type="wikidata")
        desc_element.text = selected_item.get('description')

        location_element = LET.SubElement(root, "{http://www.tei-c.org/ns/1.0}location")

        geo_element = LET.SubElement(location_element, "{http://www.tei-c.org/ns/1.0}geo", ana="wgs84")
        geo_element.text = coordinates

        # Convert the XML element to a string
        xml_snippet = LET.tostring(root, encoding='unicode')
        print(xml_snippet)

        # open register file and add root to listPlace
        register_file = os.path.join(os.getcwd(), '..', 'data', 'register', 'lassberg-places.xml')
        tree = LET.parse(register_file)
        register_root = tree.getroot()
        get_text = register_root.find('{http://www.tei-c.org/ns/1.0}text')
        get_body = get_text.find('{http://www.tei-c.org/ns/1.0}body')
        list_places = get_body.find('{http://www.tei-c.org/ns/1.0}listPlace')
        list_places.append(root)
        tree.write(register_file, encoding='utf8')

        log.log("Added to register: \n" + LET.tostring(root, encoding='unicode'))

        return xml_snippet, xml_id

    def fetch_coordinates(self, wiki_id):
        # Wikidata API URL for fetching coordinates
        api_url = f"https://www.wikidata.org/w/api.php?action=wbgetclaims&format=json&entity={wiki_id}&property=P625"

        response = requests.get(api_url)
        claims = response.json().get('claims', {})

        if 'P625' in claims:
            coordinate_claim = claims['P625'][0]  # Assuming there's only one claim
            latitude = coordinate_claim['mainsnak']['datavalue']['value']['latitude']
            longitude = coordinate_claim['mainsnak']['datavalue']['value']['longitude']
            return f"{latitude}, {longitude}"

        return "Coordinates not available"

    def create_tei_file(self):
        # write letter to file
        with open(os.path.join(os.getcwd(), '..', 'data', 'letters', f'{self.doc_title}.xml'), 'w', encoding='utf8') as f:
            f.write(self.letter_text)
                    

class TranskribusExport():
    """ This class exports page-xml files from Transkribus and saves them to the folder ../data/pagexml/documentid/."""
    def __init__(self):
        # get ids to download from cli
        self.letter_ids = self.get_numbers_from_cli()
        # get login session
        self.session = self.login(config.transcribus_user, config.transcribus_pw)
        # get documents in collection as list
        self.document_dict = self.get_document_ids_by_titles()
        # url to export page-xml
        self.list_of_urls_for_export = []
        self.export_pagexml()
        
    """ Login to Transkribus and start session

    Uses REST url https://transkribus.eu/TrpServer/rest/auth/login
    :param user: Username as string (should be specified in config.py)
    :param pw: Password as string (should be specified in config.py)
    :return: Returns session
    """

    def login(self, user, pw):
        # set session...
        session = requests.Session()
        # ..post credentials
        req = session.post('https://transkribus.eu/TrpServer/rest/auth/login',data = {'user': user, 'pw': pw})
        return session
    
    def get_document_ids_by_titles(self):
        """
        Fetches document IDs for given titles from the Transkribus collection.

        Parameters:
        letter_ids (list): A list of titles to search for.
        session (requests.Session): A session object for making HTTP requests.

        Returns:
        dict: A dict of document IDs corresponding to the given titles.
        """

        # URL construction
        query_url = f"https://transkribus.eu/TrpServer/rest/collections/{config.collection_id}/list"

        try:
            response = self.session.get(query_url)
            response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        try:
            documents = response.json()
        except (IndexError, ValueError, KeyError):
            raise ValueError("Invalid response format")

        # Create a mapping of titles to document IDs
        title_to_id = {doc['title']: doc['docId'] for doc in documents}

        # Retrieve the document IDs for the provided titles
        document_dict = {title: title_to_id.get(title) for title in self.letter_ids}
        
        return document_dict

    def export_pagexml(self):
        for doc_title, doc_data in self.document_dict.items():
            export_url = self.get_export_url(doc_data)
            self.list_of_urls_for_export.append((doc_title, doc_data, export_url))

            # download zip file to the subfolder data/pagexml...
            zip_file_name = os.path.join(os.getcwd(),'..','data','pagexml',f'{doc_title}',f'{doc_title}.zip')
            # Create the directory if it doesn't exist
            isExist = os.path.exists(os.path.join(os.getcwd(),'..','data','pagexml',f'{doc_title}'))
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(os.path.join(os.getcwd(),'..','data','pagexml',f'{doc_title}'))
            zip_file = requests.get(export_url)
            zip_file.raise_for_status()
            save_file = open(zip_file_name,'wb')
            for chunk in zip_file.iter_content(100000):
                save_file.write(chunk)
            save_file.close()

            # ...extract files in pages folder in zip file to local folder...

            destination_folder = 'destination_folder'

            with ZipFile(zip_file_name, 'r') as zip_ref:
                n = 1
                sorted_file_list = sorted(zip_ref.infolist(), key=lambda x: x.filename)
                for file_info in sorted_file_list:
                    
                    # Check if the file is in the 'pages' folder
                    if '/page/' in file_info.filename:
                        
                        # Construct new path (keeping the file's original name)
                        file_name = os.path.basename(file_info.filename)
                        new_path = os.path.join(os.getcwd(),'..','data','pagexml',f'{doc_title}',f'{doc_title}-{str(n)}.xml')
                        n += 1

                        # Extract the file
                        with open(new_path, 'wb') as output_file:
                            output_file.write(zip_ref.read(file_info.filename))

            # ...and delete zip file
            os.remove(zip_file_name)    

    def get_export_url(self, document_id):
        """ Export page-XML to local machine for further processing

        Uses REST url https://transkribus.eu/TrpServer/rest/collections/{collection-ID}/{document-ID}/fulldoc

        :param session: Transkribus session as returned from login_transkribus()
        :param collection_id: Transkribus collection number as Int
        :param document_id: Transkribus document number as Int
        :param startpage: First page of document to be exported as Int
        :param endpage: Last page of document to be exported as Int
        :return: Returns url to exported data that can be downloaded as zip-file
        """

        # concat url to document...
        url = 'https://transkribus.eu/TrpServer/rest/collections/' + str(config.collection_id) + '/' + str(document_id) + '/export'

        # ...set paramater for exporting page-xml...
        params = '{"commonPars":{"doExportDocMetadata":true,"doWriteMets":true,"doWriteImages":false,"doExportPageXml":true,"doExportAltoXml":false,"doExportSingleTxtFiles":false,"doWritePdf":false,"doWriteTei":false,"doWriteDocx":false,"doWriteOneTxt":false,"doWriteTagsXlsx":false,"doWriteTagsIob":false,"doWriteTablesXlsx":false,"doCreateTitle":false,"useVersionStatus":"Latest version","writeTextOnWordLevel":false,"doBlackening":false,"selectedTags":["add","date","Address","supplied","work","capital-rubricated","unclear","sic","structure","div","regionType","seg-supp","speech","person","gap","organization","comment","abbrev","place","rubricated"],"font":"FreeSerif","splitIntoWordsInAltoXml":false,"pageDirName":"page","fileNamePattern":"${filename}","useHttps":true,"remoteImgQuality":"orig","doOverwrite":true,"useOcrMasterDir":true,"exportTranscriptMetadata":true,"updatePageXmlImageDimensions":false},"altoPars":{"splitIntoWordsInAltoXml":false},"pdfPars":{"doPdfImagesOnly":false,"doPdfImagesPlusText":true,"doPdfWithTextPages":false,"doPdfWithTags":false,"doPdfWithArticles":false,"pdfImgQuality":"view"},"docxPars":{"doDocxWithTags":false,"doDocxPreserveLineBreaks":false,"doDocxForcePageBreaks":false,"doDocxMarkUnclear":false,"doDocxKeepAbbrevs":false,"doDocxExpandAbbrevs":false,"doDocxSubstituteAbbrevs":false}}'

        # ...post export request, starts export and returns job number...
        export_request = self.session.post(url,params)
        export_request = export_request.text

        # ...check status of job after a couple of seconds (usually, it takes around 5 seconds to export a page)...
        time.sleep(6)
        export_status = self.session.get('https://transkribus.eu/TrpServer/rest/jobs/' + export_request)
        export_status = export_status.json()

        while export_status["state"] != 'FINISHED':
            # ...check again after 5 seconds...
            export_status = self.session.get('https://transkribus.eu/TrpServer/rest/jobs/' + export_request)
            export_status = export_status.json()
            time.sleep(10)

        # ..get url of exported zip file...
        export_file_url = export_status["result"]
        return export_file_url


    def file_processing(self, file_path):
        # Implement the file processing functionality here
        pass

    def get_numbers_from_cli(self):
        ids = []
        user_input = input("Enter one or document ids (separated by spaces): ")
        numbers_str = user_input.split()
        for num_str in numbers_str:
            try:
                id = 'lassberg-letter-' + num_str.zfill(4)
                ids.append(id)
            except ValueError:
                print(f"Invalid number: {num_str}")
        return ids


class Logging():
    """ This class logs the export process to a log file. """
    def __init__(self):
        self.create_time_stamp()

    def create_time_stamp(self):
        timestamp = f"{datetime.date}".center(40, '#')
        self.log(timestamp)

    def log(self, log):
        with open(os.path.join(os.getcwd(), '..', 'logs', 'export.log'), 'a', encoding='utf8') as f:
            f.write(log)

def showcase():
    print(f"Starting export pipeline without api calls...\n".center(40, '#'))
    log = Logging()



def main():
    """ This function calls the other functions in the script to start export pipeline.
    
        1. load xml files with register data: GetDataFromXml()
        2. export page-xml files from Transkribus: TranskribusExport()
        3. iterate through documents specified in cli input and process page-xml files: ProcessPageXML()
        4. load csv file with register data: GetDataFromCsv()
        5. create xml file for each letter: CreateXML()


    """

    # Text for showcasing without calling API
    # Comment out for production
    normalized_text = """1264 163. Nr. 85. Konstanz am 30. Juli 1825. Die schöne Gelegenheit, Ihnen mein teurer Freund, durch Herrn Registrator Rosenbäcker einen freundlichen Gruß zuzurufen, will ich nicht versäumen und Ihnen sagen, dass ich letzten Montag meine Schwiegertochter, welche Ihnen mit mir noch vielmal herzlich für alle erwiesene Liebe und Freundschaft dankt, von hier nach Worblingen zu einem Freund Herrn von Soumard begleitet habe und dort in der Registratur einige interessante Urkunden fand, wovon besonders die Eine von 1444, über die Familie Holbein einen ganz unerwarteten Aufschluss gibt. Erik Holbein wird darin als der Stifter des Selhauses in Ravensburg aufgeführt. Ich hoffe Herrn Albrecht Hegner durch Mitteilung derselben einiges Vergnügen zu machen. Ich durchstrich von Worblingen aus den sogenannten Schiener Berg, besuchte die uralte Schrotzburg, in welcher vor ein Paar Jahren einige 40 römische Silbermünzen ausgegraben wurden, und fand der Sage nach, die von dem alten Hattinger in seiner Kirchengeschichte. I. Seite 482 gewagte Angabe, dass dieses die Diepoldsburg sei, wohin Berthold und Erchanger den gefangenen Bischof Salomon III verborgen haben, ganz wahrscheinlich. Ich besuchte auch Schinen, wo der allgemeinen Sage nach die ersten Christen dieses Landes, vor den Verfolgungen der Römer fliehend, sich sollen angesiedelt haben: halte aber dafür, dass es wohl möchten Leute gewesen sein, die zu Anfang des 10 Jahrhunderts, vor den alles überschwemmenden und zerstörenden Hunnen, in diesen beinahe unentdeckbaren Bergkessel sich geflüchtet haben. In Stein fand ich Herrn Pfarrer Kirchhofer abwesend und ging unausgehalten hierher zurück. Nun mein Freund! hätte ich eine kleine Bitte an den glücklichen Besitzer des Codex trad. S. Gallensium. In Neugarts Cod. diplom. Alamanniae Tom. I. Urkunde CIII., Traditis Gringi, sind ausgelassene Stellen, besonders nach den Worten: Gallone, Gringi und: Visus sum habere. Könnten Sie die Güte haben mir diese Lücken ergänzen, oder wenn die Urkunde nicht zu lange ist, lieber mir dieselbe ganz abschreiben zu lassen, so würden Sie mich recht sehr verbinden; freilich könnte nur die allergrößte Genauigkeit der Abschrift den gehörigen Wert geben. Hier lege ich Ihnen auch ein Zettelchen für den Herrn Oberst Honerlage bei, damit er sieht, dass sein Name nicht von Gestern ist, und ursprünglich dem nördlichen Deutschland angehört. Übrigens leben Sie wohl, von Eppishausen aus ein Mehreres von Ihrem Freund JvLaßzberg 1267. Konstanz, 30. Juli 1825 Josef von Lassberg beantwortet 6. Aug."""
    translated_text = """1264 163. No. 85. Konstanz, July 30th, 1825. Dear Friend, The charming opportunity to send you my best wishes through Mister Registrar Rosenbäcker, is something I won't miss. I would like to inform you that last Monday, I accompanied my daughter-in-law - who, together with me, thanks you dearly for all the love and friendship shown - from here to Worblingen to a friend, Mr. von Soumard. There, in the registry, I found some interesting documents. Particularly, one from the year 1444, providing an unexpected insight into the Holbein family. Erik Holbein is mentioned therein as the founder of Selhaus in Ravensburg. I hope to provide Mr. Albrecht Hegner some pleasure by informing him of this. From Worblingen, I traveled to the so-called Schiener Mountain, and visited the ancient Schrotzburg, where some 40 Roman silver coins were excavated a few years ago. Interestingly, I found a legend referenced by the old Hattinger in his Church History, page 482, that suggested this might be the Diepoldsburg, where Berthold and Erchanger presumably hid Bishop Salomon III – quite plausible. I also visited Schinen, which according to popular myth was purportedly the location where the first Christians of this land settled, fleeing Roman persecution – however, my belief leans more toward it being people from the beginning of the 10th Century, refugees from the destructive Hunnic invasions, who fled to these nearly undiscovered mountain hollows. I didn’t meet Pastor Kirchhofer in Stein, and uninterrupted, I returned here. Now, my dear friend, I have a small request to ask of the fortunate owner of the Codex trad. S. Gallensium. In Neugart's Cod. diplom. Alamanniae Tom. I., document CIII., Traditis Gringi, there are omitted parts, especially after the words: Gallone, Gringi, and: Visus sum habere. If you could be so kind as to fill these gaps for me or, if it is not too long, perhaps transcribe the entire document for me, you would greatly oblige me. However, only the utmost accuracy in the transcription will give it its due value. Attached, please find a note for Colonel Honerlage, so he can see that his name is both ancient and originally belongs to Northern Germany. Finally, farewell. I will have more to tell you from Eppishausen soon. Your friend, JvLaßzberg 1267. Konstanz, July 30th, 1825 Josef von Lassberg replied to on August 6th."""
    summary_text = """In einem Brief aus Konstanz vom 30. Juli 1825 berichtet Josef von Lassberg seinem Freund von seiner Reise zu einem Freund in Worblingen, wo er interessante Urkunden fand, darunter eine über die Familie Holbein. Er besuchte auch andere Orte, darunter Schiener Berg und Schrotzburg, und fand Hinweise auf historische Ereignisse. Er bittet seinen Freund um Hilfe bei der Ergänzung einer Urkunde aus dem Codex trad. S. Gallensium und legt einen Zettel für Oberst Honerlage bei, der besagt, dass sein Name ursprünglich aus dem nördlichen Deutschland stammt."""

    # 1. load xml files with register data: GetDataFromXml()
    gpt_version = 'gpt-4'

    print(f"Starting export pipeline using {gpt_version}.\n".center(20, '#'))
    
    log = Logging()
    log.log(f"\nUsing {gpt_version}")

    print("Getting data from register files.".rjust(20, '-'))

    xml_data = GetDataFromXml()

    # 2. export page-xml files from Transkribus: TranskribusExport()

    print("Logging in to Transkribus.".rjust(20, '-'))

    print("\nWaiting for user input...")

    transkribus_export = TranskribusExport()

    print("\nRecieved user input and starting to process files.")

    # 3. iterate through documents specified in cli input and process page-xml files: ProcessPageXML()
    for doc_id, _, _ in transkribus_export.list_of_urls_for_export:
        print(f"Starting export of letter {doc_id}.\n".center(20, '#'))
        log.log(f"\nStarting export of letter {doc_id}.\n".center(20, '#'))

        print("Getting metadata for letter.")
        csv_data = GetDataFromCsv(doc_id)
        
        print("Start processing pipline.")
        process_pagexml = ProcessPageXML(doc_id, gpt_version, xml_data.list_of_persons, xml_data.list_of_places, xml_data.persons_gpt_lookup, xml_data.places_gpt_lookup, log)
    
        print("Create TEI for letter.")
        xml_encoded_letter = CreateXML(doc_id, csv_data.letter_data, process_pagexml.letter_text, normalized_text, translated_text, summary_text, xml_data.list_of_persons, xml_data.list_of_places)
        #xml_encoded_letter = CreateXML(doc_id, csv_data.letter_data, process_pagexml.letter_text, process_pagexml.letter_normalization, process_pagexml.letter_translation, process_pagexml.letter_summary, xml_data.list_of_persons, xml_data.list_of_places)
    
if __name__ == "__main__":
    main()


""" Costs of openai API

Using gpt4 0.44 $ for letter 1015


"""