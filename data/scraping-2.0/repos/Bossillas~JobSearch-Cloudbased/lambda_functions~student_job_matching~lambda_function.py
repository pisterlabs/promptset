import json
import boto3
import os
import base64
import datatier
import openai
from configparser import ConfigParser
from pypdf import PdfReader



def lambda_handler(event, context):
  try:
    print("**STARTING**")
    print("**lambda: downloading job description**")

    print("_______________________")
    print("**Setup AWS**")
    # setup AWS based on config file: ==========================================
    config_file = 'config.ini'
    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = config_file
    
    configur = ConfigParser()
    configur.read(config_file)
    
    # configure for S3 access:
    s3_profile = 's3readonly'
    boto3.setup_default_session(profile_name=s3_profile)
    
    bucketname = configur.get('s3', 'bucket_name')
    
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucketname)
    
    # configure for RDS access
    rds_endpoint = configur.get('rds', 'endpoint')
    rds_portnum = int(configur.get('rds', 'port_number'))
    rds_username = configur.get('rds', 'user_name')
    rds_pwd = configur.get('rds', 'user_pwd')
    rds_dbname = configur.get('rds', 'db_name')
    openai_key = configur.get('openai', 'openai_key')





    # Get jobid from the event(from client) ====================================
    # jobid from event: could be a parameter or could be part of URL path ("pathParameters"):
    if "jobid" in event:
      jobid = event["jobid"]
    elif "pathParameters" in event:
      if "jobid" in event["pathParameters"]:
        jobid = event["pathParameters"]["jobid"]
      else:
        raise Exception("requires jobid parameter in pathParameters")
    else:
        raise Exception("requires jobid parameter in event")
        
    print("jobid:", jobid)


    # Check if jobid exists & Get job description S3 key =======================
    # open connection to the database:
    print("_______________________")
    print("**Opening connection to RDS**")
    dbConn = datatier.get_dbConn(rds_endpoint, 
                                  rds_portnum, 
                                  rds_username, 
                                  rds_pwd, 
                                  rds_dbname)

    # first check if the jobid is valid:
    print("**Checking if jobid is valid**")
    sql = "SELECT * FROM jobs WHERE id = %s;"
    row = datatier.retrieve_one_row(dbConn, sql, [jobid])
    
    # If jobid does not exist
    if row == ():  # no such job
      print("**No such job, returning...**")
      return {
        'statusCode': 400,
        'body': json.dumps("no such job...")
      }
    
    # If jobid does exist
    print("The job information: ", row)
    job_description_key = row[2]
    status = row[4]
    
    print("status:", status)
    print("job description file S3 key:", job_description_key)


    # Check the status of the job ==============================================
    # If job is closed
    if status == "closed":
      print("**Job status closed, returning...**")
      #
      return {
        'statusCode': 400,
        'body': json.dumps("Job application closed")
      }

    # If job is still open
    jd_local_filename = "/tmp/js.txt"
    
    print("_______________________")
    print("**Downloading results from S3**")
    
    full_job_description_key = "jobdescription/" + job_description_key + ".txt"
    bucket.download_file(full_job_description_key, jd_local_filename)

    # Open the downloaded file and read the contents
    with open(jd_local_filename, "r") as infile:  # Use "r" for reading text
        job_description = infile.read()
    print("**DONE reading job description**")





# Get studentid from the event(from client) ====================================
    # studentid from event: could be a parameter or could be part of URL path ("pathParameters"):
    if "studentid" in event:
      studentid = event["studentid"]
    elif "pathParameters" in event:
      if "studentid" in event["pathParameters"]:
        studentid = event["pathParameters"]["studentid"]
      else:
        raise Exception("requires studentid parameter in pathParameters")
    else:
        raise Exception("requires studentid parameter in event")
        
    print("studentid:", studentid)


    # Check if studentid exists & Get student's skill and resume S3 key ========
    # open connection to the database:
    print("_______________________")
    print("**Opening connection to RDS**")
    dbConn = datatier.get_dbConn(rds_endpoint, 
                                  rds_portnum, 
                                  rds_username, 
                                  rds_pwd, 
                                  rds_dbname)

    # first check if the studentid is valid:
    print("**Checking if studentid is valid**")
    sql = """
          SELECT 
            students.id AS id, 
            students.lastname AS lastname, 
            students.firstname AS firstname,
            students.email AS email, 
            students.linkedin AS linkedin, 
            schools.name AS schoolName,
            GROUP_CONCAT(skills.skill SEPARATOR ', ') AS skills,
            students.resume_s3 AS resume_s3
          FROM students
          JOIN schools 
            ON students.school_id = schools.id
          JOIN student_skill ss 
            ON students.id = ss.student_id
          JOIN skills 
            ON skills.id = ss.skill_id
          WHERE students.id = %s
          GROUP BY students.id, students.lastname, students.firstname, students.email, students.linkedin, schools.name, students.resume_s3
          """
    row = datatier.retrieve_one_row(dbConn, sql, [studentid])
    
    # If student does not exist
    if row == ():
      print("**No such student, returning...**")
      return {
        'statusCode': 400,
        'body': json.dumps("no such student...")
      }
    
    # If student does exist
    print("The student information: ", row)
    student_resume_key = row[7]
    skills = row[6]
    
    print("skills:", skills)
    print("student resume file S3 key:", student_resume_key)

    # Student resume local file path
    resume_local_filename = "/tmp/resume.txt"
    
    print("_______________________")
    print("**Downloading student resume from S3**")
    
    full_student_resume_key = "resume/" + student_resume_key + ".pdf"
    bucket.download_file(full_student_resume_key, resume_local_filename)

    # Open the downloaded file and read the contents
    print("**PROCESSING local PDF**")
    
    reader = PdfReader(resume_local_filename)
    number_of_pages = len(reader.pages)
    
    # Initialize an empty string to hold all text
    student_resume = ""
    
    # Iterate through each page and extract text
    for i in range(number_of_pages):
        page = reader.pages[i]
        text = page.extract_text()
        if text:  # Check if text extraction was successful
            student_resume += text + "\n"  # Append text of each page with a newline
            print("** Page", i, ", text length", len(text))
        else:
            print("** Page", i, "has no text or text extraction failed")
    
    print("**DONE reading student resume**")





    # Use GPT to give matching score of job seeker and job =====================
    print("_______________________")
    print("**Start matching score evaluation**")
    # Set the API key
    openai.api_key = openai_key
    
    # Define the system message
    system_msg = 'You are a helpful assistant who can analyze a resume and a job description to determine how well they match. Provide a matching score between 0 and 100, where 100 is a perfect match. Also, list the strengths that make the candidate a good fit for the job and areas where the candidate could improve or lacks the required skills or experience. Present your analysis in format {"score": n, "strengths": ["Strength1", "Strength2", ...], "areas_for_improvement": ["improvement1", "improvement2", ...]}'
    # Define the user message
    user_msg = f"Please analyze the following resume and job description. \n\nResume:\n" 
    
    # Make the API call
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system", 
                "content": system_msg
            },
            {
                "role": "user",
                "content": user_msg + student_resume + "\n\nJob Description:\n" + job_description + "Provide a matching score and explain the strengths and areas for improvement in the required format. Return result in one line"
            }
        ]
    )
    
    # Extract and print the response
    matching_json = response.choices[0].message.content
    print(matching_json)




    # Prepare the response
    # Since it's a text file, you can return the string directly without base64 encoding
    return {
        'statusCode': 200,
        'body': json.dumps(matching_json)  # Serialize the job description as JSON
    }
    
    
  except Exception as err:
    print("**ERROR**")
    print(str(err))
    
    return {
      'statusCode': 400,
      'body': json.dumps(str(err))
    }
