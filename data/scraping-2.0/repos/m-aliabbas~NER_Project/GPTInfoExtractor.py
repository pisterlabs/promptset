import docx2txt
import pandas as pd
import openai

class GPTInfoExtractor(object):
    """
    This is GPT Based Infor Extractor for Human text in word docs
    """
    def __init__(self,api_key) -> None:
        self.text_doc = ""
        self.defualt_attr_db = pd.read_csv('default_enum_db.csv')
        self.attr_list = list(self.defualt_attr_db['Data'].values)
        self.example_sub_dict = {'value':'some_value','start_index':'starting_index','end_index':'ending_index'}
        self.prompt = f""" {self.text_doc} Please extract these {self.attr_list} from above text;
            Also extract starting and ending index of entity  with each key in subdictionary and return Python dictionary. 
            Subdictionary will have like this {self.example_sub_dict} or None
            please ignore the entities if their value is none or empty.
            """
        self.attributes = []
        self.attribute_db = {}
        self.api_key = api_key
        openai.api_key = api_key
        
    def __extract_attribute(self, text):
        self.text_doc = text
        self.attribute_db = {}
        self.prompt = f""" {self.text_doc} Please extract these {self.attr_list} from above text;
            Also extract starting and ending index of entity  with each key in subdictionary and return Python dictionary. 
            Subdictionary will have like this {self.example_sub_dict} or None
            please ignore the entities if their value is none or empty.
            """
        
        try:
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"{self.prompt}"}])
            content = completion.choices[0].message.content
            self.attribute_db = eval(content)
            # print('Here is attributed db',self.attribute_db)
        except Exception as e:
            print(f'Following Error Occured:\n {e}')
            pass
    
    def __extract_name(self, text):
        """
        Extract first and last name from text and append to attributes list.
        """

        text1 = text.lower()
        if ('hey' in text1) or ('hi' in text1) or ('hello' in text1):
            text = text.split(" ")
            try:
                first_name = text[1]
                last_name = text[-1]
            except IndexError:
                first_name, last_name = text[-1], text[-1]

            
            # Get the start and ending indices of first and last names in text1
            first_name_start = text1.index(first_name.lower())
            first_name_end = first_name_start + len(first_name)
            last_name_start = text1.index(last_name.lower())
            last_name_end = last_name_start + len(last_name)
            
            # # Append the start and ending indices to the attributes list
            # self.attribute_db['First Name'] = first_name_start
            # self.attribute_db['First Name End'] = first_name_end
            # self.attribute_db['Last Name Start'] = last_name_start
            # self.attribute_db['Last Name End'] = last_name_end
            
            self.attribute_db['First Name'] = {'value':first_name,'start_index':first_name_start,'end_index':first_name_end}
            self.attribute_db['Last Name'] = {'value':last_name,'start_index':last_name_start,'end_index':last_name_end}


    def __del_gpt_full_name(self):
        try:
            del self.attribute_db['Name']
        except:
            pass

    def get_attributes(self, text):

        """
        Split input text into lines and extract attributes and names from each line.
        Return a dictionary containing extracted attributes.
        """
        self.__extract_attribute(text)
        self.text_lines = text.split('\n')
        for index, text_line in enumerate(self.text_lines):
            if index < 3:
                self.__extract_name(text=text_line)
        self.__del_gpt_full_name()
        # print(self.attribute_db)
        # self.attribute_db={'First Name': {'value': 'Evan', 'start_index': 4, 'end_index': 8}, 'Middle Name': None, 'Last Name': {'value': 'Zigomalas,', 'start_index': 9, 'end_index': 19}, 'Suffix': None, 'Salutation': None, 'Mobile Phone Number': None, 'Mobile Phone Number (Work)': None, 'Date of Birth': None, 'Place of Birth': None, 'Country of Birth': None, 'Mother’s Maiden Name': None, 'Nationality': None, 'Passport Number': {'value': '337808142', 'start_index': 22, 'end_index': 31}, 'Passport Authority': None, 'Passport Issue Date': None, 'Passport Expiry Date': None, 'Driver’s License Number': {'value': 'Zigoms789123evjv', 'start_index': 116, 'end_index': 131}, 'Driver’s License Expiry Date': None, 'Social Security Number / National Insurance Number': {'value': 'ET 91 21 81 C', 'start_index': 39, 'end_index': 51}, 'Home Address Line 1': '5 Binney St', 'Home Address Line 2': None, 'Home Address Town': None, 'Home Address City': None, 'Home Address Zip / Postcode': {'value': 'HP11 2AX', 'start_index': 152, 'end_index': 160}, 'Home Address County': None, 'Home Address State': None, 'Work Company Name': None, 'Work Job Title': None, 'Work Start Date': None, 'Email Address': {'value': 'evan.zigomalas@gmail.com', 'start_index': 167, 'end_index': 192}, 'Telephone Number': {'value': '01937-864715', 'start_index': 194, 'end_index': 207}, 'Work Address ': None, 'Work Address Line 2': None, 'Work Address Town': None, 'Work Address City': None, 'Work Address County': None, 'Work Address State': None, 'Work Address Zip / Postcode': None, 'Work Address Country': None, 'Health Security Number (NHS for the UK)': {'value': '512 880 3880', 'start_index': 139, 'end_index': 151}, 'TAX ID': None, 'State Identification Number': None, 'Bank Account Name': None, 'Bank Account Number': {'value': '83012372', 'start_index': 77, 'end_index': 85}, 'Bank Account Sort Code': {'value': '20-45-07', 'start_index': 87, 'end_index': 94}, 'Bank Account Routing number': None, 'Next of Kin Name': None, 'Next of Kin Relationship': None, 'Next of Kin Contact Number': None, 'Race': None, 'Religion': None, 'Sex at Birth': None, 'Sex Now': None, 'Pronouns': None, 'Eye Colour': None, 'DNA File': None, 'Doctors Name': None, 'Doctors Address': None, 'Doctors Address 2': None, 'Doctors Address Town': None, 'Doctors Address ZIP / Postcode': None, 'Bitcoin Address': None, 'Ethereum Address': None, 'Facebook': None, 'Instagram': None, 'Twitter': None, 'Company Name': None, 'Company Contact Name': None, 'Company VAT  / TAX ID': None, 'Company Registration ID': None, 'DUNS ID': None, 'SIC Code': None, 'Credit Card Number': {'value': '5602246091873661', 'start_index': 103, 'end_index': 119}}
        return self.attribute_db
    


# info_extractor = GPTInfoExtractor(api_key="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")
# text = docx2txt.process("/home/ali/Desktop/waspak_co/NER_Project/test/Marg Grasmick.docx")
# attr=info_extractor.get_attributes(text=text)
# info_extractor.save_csv('/home/ali/Desktop/waspak_co/NER_Project/output/Marg Grasmick.csv')
# print(attr)
