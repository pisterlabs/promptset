import ast
import glob
import os
from pprint import pprint
from time import sleep

import PyPDF2.errors
from openai import OpenAI
from pdf2image import convert_from_path
from simpleaichat import AIChat
from tqdm import tqdm

from database import IncidentsFromPdf, get_database_session, Persons

from PyPDF2 import PdfFileWriter, PdfReader, PdfWriter
import pytesseract

from police_fire.maps.get_lat_lng_of_addresses import get_lat_lng_of_address
from police_fire.utilities import check_if_details_references_a_relative_date, \
    check_if_details_references_an_actual_date, get_incident_location_from_details \
    , add_or_get_person

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
DBsession, engine = get_database_session(environment='prod')

ai = AIChat(
    console=False,
    save_messages=False,  # with schema I/O, messages are never saved
    model="gpt-4",
    params={"temperature": 0.0},
    api_key=os.getenv('OPENAI_API_KEY'),
)

month_str_to_int = {
    'jan': '01',
    'feb': '02',
    'mar': '03',
    'apr': '04',
    'may': '05',
    'jun': '06',
    'jul': '07',
    'aug': '08',
    'sep': '09',
    'oct': '10',
    'nov': '11',
    'dec': '12'
}


def get_pdf_path(year, month, day):
    pdf_glob_path = rf'C:\Users\Nick\PycharmProjects\cortlandStandardScraper\pdfs\{year}\{month}\{day}\*'
    try:
        path_results = glob.glob(pdf_glob_path)
        pdf_path = [path_result for path_result in path_results if path_result.endswith('.pdf')][0]
        print('pdf_path:', pdf_path)
        return pdf_path
    except IndexError:
        print(f'No PDF found for {year}-{month}-{day}.')
        return


def convert_newspaper_page_to_text(input_pdf, newspaper_page_number, pages_path):
    #print('newspaper_page_number:', newspaper_page_number)
    output = PdfWriter()
    try:
        page_object = input_pdf.pages[newspaper_page_number]
    except IndexError:
        print(f'No page {str(newspaper_page_number)} found.  Not adding to database.')
        return
    output.add_page(page_object)

    policeFirePagePath = pages_path + '\\' + "police_fire_page.pdf"
    with open(policeFirePagePath, "wb") as output_stream:
        output.write(output_stream)

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    page_data = convert_from_path(policeFirePagePath,
                                  poppler_path=r'C:\Users\Nick\Downloads\Release-23.11.0-0\poppler-23.11.0\Library\bin')
    page_data = page_data[0]

    #print('newspaper_page_number:', newspaper_page_number)
    txt = pytesseract.pytesseract.image_to_string(page_data).encode("utf-8")
    return txt


def scrape_police_fire_data_from_pdf(pdf_path, year_month_day_str):
    #print(pdf_path)
    pages_path = '\\'.join(pdf_path.split('\\')[:-1]) + '\\pages'
    if not os.path.exists(pages_path):
        os.mkdir(pages_path)

    try:
        input_pdf = PdfReader(pdf_path)
    except PermissionError as e:
        raise e
    except PyPDF2.errors.PdfReadError:
        raise Exception(f'Could not read PDF: {pdf_path}')
        return
    for newspaper_page_number in [2, 1]:
        # it's more often the case that the police/fire details are on the 3rd page of the PDF,
        # but sometimes they're on the 2nd page.
        txt = convert_newspaper_page_to_text(input_pdf, newspaper_page_number, pages_path)
        txt = str(txt)
        with open(pages_path + '\\' + f'police_fire_page_{str(newspaper_page_number)}.txt', 'w') as f:
            f.write(txt)
        raw_incidents = ['Accused:' + i for i in txt.split('Accused:')][1:]
        query = ("List all of the incident details provided in the following string, in the original language"
                 " of the article.  Use a Python-style dictionaries with the keys \'accused_name\', \'accused_age\',"
                 " \'accused_location\', \'charges\', \'details\', \'legal_actions\'. All values need to be strings. "
                 "If the article is not about a crime, the output should be N/A.")

        for raw_incident in raw_incidents:
            #print(raw_incident)
            query_with_incident = query + raw_incident
            response = ai(query_with_incident)
            response_as_dict = ast.literal_eval(response)
            if type(response_as_dict) is list:
                for incident in response_as_dict:
                    if incident == 'N/A':
                        print('No police/fire details found.  Not adding to database.')
                        continue
                    else:
                        parse_details_for_incident(incident, year_month_day_str)
            elif type(response_as_dict) is dict:
                parse_details_for_incident(response_as_dict, year_month_day_str)
            elif response_as_dict == 'N/A':
                continue
            else:
                raise ValueError(f'Unexpected response: {response_as_dict}')
        # if we've gotten this far, we've successfully scraped the police/fire details from the first page of the PDF
        # and there's no need to continue the loop to check the 3rd page.
        break

    return


def parse_details_for_incident(incident, year_month_day_str):
    pprint(incident)
    incident_date_response = check_if_details_references_a_relative_date(incident['details'],
                                                                         year_month_day_str)
    incident_location = get_incident_location_from_details(incident['details'])
    if not incident_date_response:
        # check if details references an actual date
        incident_date_response = check_if_details_references_an_actual_date(incident['details'],
                                                                            year_month_day_str)
    existing_incident = DBsession.query(IncidentsFromPdf).filter(
        IncidentsFromPdf.incident_reported_date == year_month_day_str,
        IncidentsFromPdf.accused_name == incident['accused_name'],
        IncidentsFromPdf.accused_age == incident['accused_age'],
    ).all()
    lat, lng = get_lat_lng_of_address(incident_location)
    existing_incident = existing_incident[0] if len(existing_incident) > 0 else None
    if existing_incident:
        print('Existing incident found.  Not adding to database.')
        return
    else:
        incidentFromPdf = IncidentsFromPdf(
            incident_reported_date=year_month_day_str,
            accused_name=incident['accused_name'],
            accused_age=incident['accused_age'],
            accused_location=incident['accused_location'],
            charges=incident['charges'],
            details=incident['details'],
            legal_actions=incident['legal_actions'],
            incident_date=incident_date_response,
            incident_location=incident_location,
            incident_location_lat=lat,
            incident_location_lng=lng,
            accused_person_id=add_or_get_person(DBsession, incident['accused_name'])
        )
        DBsession.add(incidentFromPdf)
        DBsession.commit()
        sleep(1)

    return


def main():
    years = [
        #'2017',
        #'2018',
        #'2019',
        #'2020',
        '2021',
        #'2022'
    ]
    months = [
        #'jan',
        #'feb',
        'mar',
        #'apr',
        #'may',
        #'jun',
        #'jul'
        #'aug',
        #'sep',
        #'oct',
        #'nov',
        #'dec',
    ]
    day_numbers = [str(day_number) for day_number in range(1, 32)]
    for year in tqdm(years, desc='year'):
        for month in tqdm(months, desc='month'):
            for day_number in tqdm(day_numbers, desc='day'):
                year_month_day_str = f'{year}-{month}-{day_number}'
                pdf_path = get_pdf_path(year, month, day_number)
                if not pdf_path:
                    continue
                else:
                    scrape_police_fire_data_from_pdf(pdf_path, year_month_day_str)
    return


if __name__ == '__main__':
    main()
