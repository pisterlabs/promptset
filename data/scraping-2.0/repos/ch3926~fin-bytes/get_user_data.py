import gspread
import json
import csv
import smtplib
import ssl

import news_api
import openai_api

sa = gspread.service_account(filename="fin-bytes.json") # service account to use for api
sh = sa.open("fin-bytes") # the google sheet containing all the user entered data
sheet = sh.worksheet("Investing Survey") # specific sheet containing the data

# grab user entered records from spreadsheet so we can use them here
sheet_dict = sheet.get_all_records()
sheet_list = sheet.get_all_values()

#print(sheet_dict)
#print(sheet_list)

# convert list of dict values into json
# json_info = json.dumps(sheet_dict)
# print(json_info)

# convert list of dicts to csv

keys = sheet_dict[0].keys()

with open('user_info.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(sheet_dict)


def send_mail(news, summary):
    message = """Subject: Weekly Fin Byte!

    Hi {firstName} {lastName}, here is your weekly update on {interests} 

    {summary}

    {news}
    """

    from_address = "finbytes.info@gmail.com"
    email_password = "ytge xsnj zrnl xigg"

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(from_address, email_password)

        with open("user_info.csv") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for user in reader:
                first_name = user[0]
                last_name = user[1]
                email = user[2]
                level = user[3]
                interests = user[4]
                date_time = user[5]

                server.sendmail(
                    from_address,
                    email,
                    message.format(firstName=first_name,lastName=last_name, interests=interests),
            )

send_mail()