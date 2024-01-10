import json
import pandas as pd
import openai
import pandas as pd
from database.ConnectDb import DatabaseHandler
from datetime import datetime
# from utils import get_pdf_file_names_from_db
# from src.receiveResponse.fetch_mail import pipeline
# from src.receiveResponse.fetch_mail_api import pipeline
import re
db = DatabaseHandler()

from datetime import datetime,timedelta

def extract_and_concat_description(data):
    # Extract 'Description of work done' values without 'None'
    descriptions = [row['Description of work done'] for row in data['values'] if row['Description of work done']]
    typeofwork =[row['Type of Work'] for row in data['values'] if row['Type of Work']]
    allocatehours =[row['Allocate Hours'] for row in data['values'] if row['Allocate Hours']]

    
    # Join the values into a comma-separated string
    concatenated_description = ', '.join(descriptions)
    concatenated_typeofwork = ', '.join(typeofwork)
    concatenated_allocatehours = ', '.join(allocatehours)
    
    return concatenated_description,concatenated_typeofwork,concatenated_allocatehours

def process_and_update_database(reply_data,subject,sender,df):
    try:
        print("Process and Update Database")
        pattern  = r'(?P<projectDescription>.+)-(?P<timesheetEffort>\d+\.\d+)'
        # pattern = r'^Re:\s+(?P<projectDescription>.+)-(?P<timesheetEffort>\d+\.\d+)'
        # Use re.search to find the matches
        match = re.search(pattern, str(subject))
        # Check if a match was found
        print(match)
        if match:               
            project_description = match.group('projectDescription').strip()
            project_description = re.sub(r'^\d+\s*Re:', '', project_description).strip()
            project_description = re.sub(r'^Re:\s*', '', project_description).strip()
            # project_description = project_description.lstrip('Re: ').strip()
            timesheet_effort = match.group('timesheetEffort')
            try:
                query1 = """ SELECT                
                    spocMailid,
                    spocName,
                    timeSheetEffort,
                    projectDescription,
                    taskType
                FROM
                    naggingDetails
                WHERE
                    timesheetEffort IS NOT NULL
                    AND projectDescription IS NOT NULL
                    AND taskType = 'Pending'
                    AND (projectDescription LIKE '%{project_description}%')
                """.format(project_description=project_description)

                current_timestamp = datetime.now()
                data = db.fetch_data(query1)
                if not data.empty:
                    try:
                        for index, row in data.iterrows():
                            spocMailid = row['spocMailid']
                            spocMailid = 'mukesh.kumar@algo8.ai'
                            timeSheetEffort = row['timeSheetEffort']
                            projectDescription = row['projectDescription']
                            taskType = row['taskType']
                            receiver_mail = sender
                            if (spocMailid == sender) and (float(timeSheetEffort) == float(timesheet_effort)): 
                                repliedByEmployee,concatenated_typeofwork,concatenated_allocatehours = extract_and_concat_description(reply_data)
                                toRecipients = df['toRecipients'][0]
                                ccRecipients = df['ccRecipients'][0]
                                update_dbQuery = """
                                    UPDATE naggingDetails
                                    SET Response = '{response}'
                                    WHERE spocMailid = '{spocMailid}'
                                    AND timeSheetEffort = {timeSheetEffort}
                                    AND projectDescription = '{projectDescription}'
                                    AND taskType = '{tasktype}'
                                    AND toRecipients = '{toRecipients}'
                                    AND ccRecipients = '{ccRecipients}'
                                    AND Subject = '{Subject}'
                                """.format(response=repliedByEmployee, spocMailid=sender, timeSheetEffort=timesheet_effort,projectDescription=project_description,tasktype='Pending',toRecipients=toRecipients,ccRecipients=ccRecipients,Subject=subject)
                                # print(update_dbQuery)
                                db.execute_query(update_dbQuery)
                    except Exception as e:
                        print(e)
                
            except Exception as e:
                print("Error",e)
        else:
            print("No match found.")

    except Exception as e:
        print(e)
# taskDescription = received_data['repliedByEmployee']

# for task in taskDescription:
#     feedback =  "RND" #fetchCompletionData(task)
#     received_data['gptFeedback'] = feedback
# # print(received_data)

    # db.run_df('nagTable',received_data)

# Calculate the start and end dates and times
# current_date = datetime.now()
# # endDate = current_date.strftime('%Y-%m-%d')
# # startDate = (current_date - timedelta(days=3)).strftime('%Y-%m-%d')
# startDate = "2023-10-24"
# endDate = "2023-10-24"
# startTime = "00:00:00"
# endTime = "23:59:00"
# start_date_filter = datetime.strptime(startDate, '%Y-%m-%d').date()
# end_date_filter = datetime.strptime(endDate, '%Y-%m-%d').date()

# start_time_filter = datetime.strptime(startTime, '%H:%M:%S').time()
# end_time_filter = datetime.strptime(endTime, '%H:%M:%S').time()

# process_and_update_database(start_date_filter, end_date_filter, start_time_filter, end_time_filter)