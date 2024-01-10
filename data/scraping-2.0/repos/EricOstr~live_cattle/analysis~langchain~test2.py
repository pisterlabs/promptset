

from dotenv import load_dotenv
import sys
sys.path.append('../')

load_dotenv()

import openai

import os
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.text_splitter import CharacterTextSplitter

from reports import report1


release_date = None
month = None


tables = {
    'commercial_red_meat_production_united_states': {
        'name_in_report': 'Commercial Red Meat Production - United States',
        'rows': {
            'beef': {'name_in_report': 'Beef', 'value': None},
            'veal': {'name_in_report': 'Veal', 'value': None},
            'pork': {'name_in_report': 'Pork', 'value': None},
            'lamb_and_mutton': {'name_in_report': 'Lamb and Mutton', 'value': None},
        },
        'table_in_report' : ''
    },

    'federally_inspected_red_meat_production': {
        'name_in_report': 'Federally Inspected Red Meat Production - United States',
        'rows': {
            'beef': {'name_in_report': 'Beef', 'value': None},
            'veal': {'name_in_report': 'Veal', 'value': None},
            'pork': {'name_in_report': 'Pork', 'value': None},
            'lamb_and_mutton': {'name_in_report': 'Lamb and Mutton', 'value': None},
        },
        'table_in_report' : ''
    },

    'livestock_slaughter_and_average_live_weight': {

        'name_in_report': 'Livestock Slaughter and Average Live Weight - United States',
        'rows': {
            'cattle_number_of_head_federally_inspected': {'name_in_report': 'Cattle Number of Head Federally Inspected', 'value': None},
            'cattle_number_of_head_other': {'name_in_report': 'Cattle Number of Head Other', 'value': None},
            'cattle_number_of_head_commercial': {'name_in_report': 'Cattle Number of Head Commercial', 'value': None},
            'cattle_average_live_weight_federally_inspected': {'name_in_report': 'Cattle Average Live Weight Federally Inspected', 'value': None},
            'cattle_average_live_weight_other': {'name_in_report': 'Cattle Average Live Weight Other', 'value': None},
            'cattle_average_live_weight_commercial': {'name_in_report': 'Cattle Average Live Weight Commercial', 'value': None},

            'calves_number_of_head_federally_inspected': {'name_in_report': 'Calves Number of Head Federally Inspected', 'value': None},
            'calves_number_of_head_other': {'name_in_report': 'Calves Number of Head Other', 'value': None},
            'calves_number_of_head_commercial': {'name_in_report': 'Calves Number of Head Commercial', 'value': None},
            'calves_average_live_weight_federally_inspected': {'name_in_report': 'Calves Average Live Weight Federally Inspected', 'value': None},
            'calves_average_live_weight_other': {'name_in_report': 'Calves Average Live Weight Other', 'value': None},
            'calves_average_live_weight_commercial': {'name_in_report': 'Calves Average Live Weight Commercial', 'value': None},

            'hogs_number_of_head_federally_inspected': {'name_in_report': 'Hogs Number of Head Federally Inspected', 'value': None},
            'hogs_number_of_head_other': {'name_in_report': 'Hogs Number of Head Other', 'value': None},
            'hogs_number_of_head_commercial': {'name_in_report': 'Hogs Number of Head Commercial', 'value': None},
            'hogs_average_live_weight_federally_inspected': {'name_in_report': 'Hogs Average Live Weight Federally Inspected', 'value': None},
            'hogs_average_live_weight_other': {'name_in_report': 'Hogs Average Live Weight Other', 'value': None},
            'hogs_average_live_weight_commercial': {'name_in_report': 'Hogs Average Live Weight Commercial', 'value': None},                    

            'goats_number_of_head_federally_inspected': {'name_in_report': 'Goats Number of Head Federally Inspected', 'value': None},
            'goats_number_of_head_other': {'name_in_report': 'Goats Number of Head Other', 'value': None},
            'goats_number_of_head_commercial': {'name_in_report': 'Goats Number of Head Commercial', 'value': None},
            'goats_average_live_weight_federally_inspected': {'name_in_report': 'Goats Average Live Weight Federally Inspected', 'value': None},
            'goats_average_live_weight_other': {'name_in_report': 'Goats Average Live Weight Other', 'value': None},
            'goats_average_live_weight_commercial': {'name_in_report': 'Goats Average Live Weight Commercial', 'value': None},

            'bison_number_of_head_federally_inspected': {'name_in_report': 'Bison Number of Head Federally Inspected', 'value': None},
            'bison_number_of_head_other': {'name_in_report': 'Bison Number of Head Other', 'value': None},
            'bison_number_of_head_commercial': {'name_in_report': 'Bison Number of Head Commercial', 'value': None},
            'bison_average_live_weight_federally_inspected': {'name_in_report': 'Bison Average Live Weight Federally Inspected', 'value': None},
            'bison_average_live_weight_other': {'name_in_report': 'Bison Average Live Weight Other', 'value': None},
            'bison_average_live_weight_commercial': {'name_in_report': 'Bison Average Live Weight Commercial', 'value': None},            
        },
        'table_in_report' : ''
    },


    'livestock_slaughtered_under_federal_inspection_by_class': {

        'name_in_report': 'Livestock Slaughtered Under Federal Inspection by Class - United States',
        'rows': {
            'cattle_steers': {'name_in_report': 'Cattle Steers', 'value': None},
            'cattle_heifers': {'name_in_report': 'Cattle Heifers', 'value': None},
            'cattle_all_cows': {'name_in_report': 'Cattle All Cows', 'value': None},
            'cattle_dairy_cows': {'name_in_report': 'Cattle Dairy Cows', 'value': None},
            'cattle_other_cows': {'name_in_report': 'Cattle Other Cows', 'value': None},
            'cattle_bulls': {'name_in_report': 'Cattle Bulls', 'value': None},

            'calves_and_vealers': {'name_in_report': 'Calves and Vealers', 'value': None},

            'hogs_barrows_and_gilts': {'name_in_report': 'Hogs Barrows and Gilts', 'value': None},
            'hogs_sows': {'name_in_report': 'Hogs Sows', 'value': None},
            'hogs_boars': {'name_in_report': 'Hogs Boars', 'value': None},

            'sheep_mature_sheep': {'name_in_report': 'Sheep Mature Sheep', 'value': None},
            'sheep_lambs_and_yearlings': {'name_in_report': 'Sheep Lambs and Yearlings', 'value': None},

        },
        'table_in_report' : ''
    },


    'federally_inspected_slaughtered_average_dressed_weight_by_class': {

        'name_in_report': 'Federally Inspected Slaughter Average Dressed Weight by Class - United States',
        'rows': {
            'cattle_steers': {'name_in_report': 'Cattle Steers', 'value': None},
            'cattle_heifers': {'name_in_report': 'Cattle Heifers', 'value': None},
            'cattle_all_cows': {'name_in_report': 'Cattle All Cows', 'value': None},
            'cattle_bulls': {'name_in_report': 'Cattle Bulls', 'value': None},

            'calves_and_vealers': {'name_in_report': 'Calves and Vealers', 'value': None},

            'hogs_barrows_and_gilts': {'name_in_report': 'Hogs Barrows and Gilts', 'value': None},
            'hogs_sows': {'name_in_report': 'Hogs Sows', 'value': None},
            'hogs_boars': {'name_in_report': 'Hogs Boars', 'value': None},

            'sheep_mature_sheep': {'name_in_report': 'Sheep Mature Sheep', 'value': None},
            'sheep_lambs_and_yearlings': {'name_in_report': 'Sheep Lambs and Yearlings', 'value': None},

        },
        'table_in_report' : ''
    },




}

prompts = {
    'get_release_date_and_month':
    '''
    From the previously sent text, find the release date of the report, and the \
    latest month of concern the report contains information about. Return the \
    release date and month in a tuple, where each element is in string format of \
    "%Y-%m-%d". The first element is a string representing the release date, and \
    the second element is a string representing the reporting month with the day (%d) 
    set to the first day of that month of that year. For example, if you identify that
    the release date of a report is July 20 2023, and the reporting month was June 2023,
    then your output should be ('2023-07-20', '2023-06-01'). Your response to this
    message should be strictly only text, representing a python tuple, which can be
    directly used in futher python code.
    ''',

    'is_table_present': '''Does the previously given text contain a table with the exact name {} which
     data in rows and columns. Note that the table's name may appear in an index or in other text, but this does not mean that the table iteself is present in the text.
     
     
     '''
}



####################


def get_date_info():

    from datetime import datetime
    import ast

    messages = [{'role':'user', 'content': report1[:3000]}]
    messages.append({'role':'user', 'content': prompts['get_release_date_and_month']})

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages
    ).choices[0].message.content

    dates_str = ast.literal_eval(response)

    release_date, month = datetime.strptime(dates_str[0], '%Y-%m-%d').date(), datetime.strptime(dates_str[1], '%Y-%m-%d').date()


    return release_date, month



release_date, month = get_date_info()


# Get table as strings
current_table = None
count_lines = 0

for line in report1.splitlines():

    for table in tables:
        if tables[table]['name_in_report'] == line:
            current_table = table
            count_lines = 0


    if current_table:
        tables[current_table]['table_in_report'] += line + '\n'

    if line.startswith('---'):
        count_lines += 1
    
    if count_lines == 3:
        current_table = None
        count_lines = 0




def clean_numeric_string(s):
    if ',' in s:
        s = s.replace(',', '')
    return s





table_info = tables['federally_inspected_slaughtered_average_dressed_weight_by_class']

messages = [{'role':'user','content': table_info['table_in_report']}]
messages.append({'role':'system',
    'content': f'Return the previously given table, with only the column "{month.strftime("%B %Y")}". DO NOT REUTRN ANYTHING ELSE, EXCEPT FOR THE NEW TABLE'
})

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=messages
).choices[0].message.content




for table, table_info in tables.items():

    # First extract the right column from the table
    messages1 = [{'role':'user','content': table_info['table_in_report']}]
    messages1.append({'role':'system',
        'content': f'Return the previously given table, with the original row names, but only the third column "{month.strftime("%B %Y")}". DO NOT REUTRN ANYTHING ELSE, EXCEPT FOR THE NEW TABLE'
    })
    
    extracted_table = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages1
    ).choices[0].message.content


    for row, row_info in table_info['rows'].items():

        messages2 = [{'role':'user','content': extracted_table}]
        messages2.append({'role':'system', 'content': f'From the previously given table, find the exact numeric value of "{row_info["name_in_report"]}". Only return the exact raw numeric value and nothing else, so that the rest of the python code can parse this output directly. ONLY REPLY WITH A NUMERIC VALUE, NOT A STRING'})
        messages2.append({'role':'system', 'content': f'YOUR OUTPUT CAN ONLY BE A NUMERIC VALUE, NO TEXT'})

        extracted_value = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages2
        ).choices[0].message.content


        row_info['value'] = float(clean_numeric_string(extracted_value))


pass


