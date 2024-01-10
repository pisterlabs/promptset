import json
import boto3
import os
import base64
import datatier
import openai
from configparser import ConfigParser



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
    local_filename = "/tmp/results.txt"
    
    print("_______________________")
    print("**Downloading results from S3**")
    
    full_job_description_key = "jobdescription/" + job_description_key + ".txt"
    bucket.download_file(full_job_description_key, local_filename)

    # Open the downloaded file and read the contents
    with open(local_filename, "r") as infile:  # Use "r" for reading text
        job_description = infile.read()
    print("**DONE reading job description**")


    # Use GPT to extract key words from job ====================================
    print("_______________________")
    print("**Start key word extraction**")
    # Set the API key
    openai.api_key = openai_key
    
    # Define the system message
    system_msg = 'You are a helpful assistant who help extract information from paragraphs and give result in JSON format'
    # Define the user message
    user_msg = "From the job description, check if the job provides sponsorship(required to be US citizen/green card holder. If not mentioned then put the value 'NA'), check if requires security clearance, and extract all the skills required for the job and number of years of working experience required(if not mentioned then put -1 for the value, if mentioned multiple time, then use the largest value) and education background requirement for the job and return in format: {'provide sponsorship': 'Yes'/'No/NA', 'security clearance':'Yes'/'No','skills': [skill1, skill2, ...], 'number of years of working experience':n, 'education background requirement':[requirement1, requirement2]. Example answer: {'provide sponsorship': 'No', 'security clearance':'Yes', 'skills': ['Python', 'SQL', 'PySpark', 'Time Series'...], 'number of years of working experience':5, 'education background requirement':['Masters degree in quantitative field or MBA', 'Bachlors degree in quantitative field']}. Do not include any other words other than the list of skills and experience. Do not include 's in the result, and return in one line: "
    
    response = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = [
                {
                    "role": "system", 
                    "content": system_msg
                },
                {
                    "role": "user",
                    "content": user_msg + job_description,
                }
        ]
    )
    
    jd_json = response.choices[0].message.content
    print(jd_json)



    # Prepare the response
    # Since it's a text file, you can return the string directly without base64 encoding
    return {
        'statusCode': 200,
        'body': json.dumps(jd_json)  # Serialize the job description as JSON
    }
    
    
  except Exception as err:
    print("**ERROR**")
    print(str(err))
    
    return {
      'statusCode': 400,
      'body': json.dumps(str(err))
    }
