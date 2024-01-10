
import os
import datetime
import requests
from report_text import openai_mail, openai_key, openai_image
from report_page import create_report_cover
import variables

def add_mail_main(mail_main, analysis_YearMonth):
    mail_head = "Dear [Recipient],"
    mail_info = variables.email_head.format(analysis_YearMonth=analysis_YearMonth)
    mail_end = variables.email_end
    mail_signature = variables.email_sign
    mail_time = datetime.datetime.now().strftime("%Y-%m-%d")
    out_content = mail_head + "\n\n" + mail_info + "\n\n" + mail_main + "\n\n" + mail_end + "\n\n" + mail_signature + "\n\n" + mail_time + "\n\n"
    return out_content

def openai_mail_cover(table_data_str, table_legend, analysis_YearMonth):
    mail_content = openai_mail(os.environ['MAIL_MAIN_CREATE'],
                               os.environ['MAIL_MAIN_CHECK'],
                               variables.mail_create.format(table_data_str=table_data_str, table_legend=table_legend, analysis_YearMonth=analysis_YearMonth),
                               variables.mail_check)
    key_words = openai_key(os.environ['MAIL_KEYWORDS_CREATE'],
                           os.environ['MAIL_KEYWORDS_CHECK'],
                           variables.key_create.format(mail_content=mail_content),
                           variables.key_check)
    print(key_words)

    image_url = openai_image(os.environ['REPORT_COVER_CREATE'], key_words, variables.cover_image)
    response = requests.get(image_url)

    if response.status_code == 200:
        with open("./temp/cover.jpg", "wb") as file:
            file.write(response.content)
    else:
        print("Error: Failed to download the image.")
    mail_content = add_mail_main(mail_content, analysis_YearMonth)
    return mail_content

def create_cover_mail(table_data_str, table_legend, analysis_YearMonth, analysis_MonthYear):
    mail_content = openai_mail_cover(table_data_str, table_legend, analysis_YearMonth)
    with open(f'../Report/mail/{analysis_YearMonth}.md', 'w') as f:
        f.write(mail_content)
    try:
        create_report_cover(analysis_MonthYear)
        print("Created the cover image successfully.")
    except:
        print("Error: Failed to create the cover image.")

def create_mail_table(table_data, analysis_YearMonth):
    table_data_str = table_data.to_markdown(index=False)    
    with open(f'../Report/table/{analysis_YearMonth}.md', 'w') as f:
        f.write(table_data_str)