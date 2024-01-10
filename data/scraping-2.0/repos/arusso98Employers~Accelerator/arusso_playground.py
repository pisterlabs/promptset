import snowflake.connector
import requests
from bs4 import BeautifulSoup
import csv
import openai
import os
import sys

#########################################################################################################
# GPT API THE 3.0 VERSION WHICH IS NOT CAPABLE OF SCRAPING AND IS NOT READY FOR THIS TASK

#set up your OpenAI API key as an environment variable
#os.environ["OPENAI_API_KEY"] = "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"
#openai.api_key = os.getenv("OPENAI_API_KEY")

def get_gpt_results(integrationtool):
    # define the prompt you want to send to the API
    prompt = "list all the applications that Azure Data Factory can connect to?"

    # set up the request parameters
    model_engine = "davinci"
    temperature = 0.1
    max_tokens = 300

    # send the prompt to the API and get the response
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # print the response from the API
    print(response.choices[0].text)


####################################################################################################
# ESTABLISH SNOWFLAKE CONNECTION

# Set up connection parameters 
account = 'strivepartner.east-us-2.azure'
user = 'arusso'
password = 'London01!'
database = 'ACCELERATOR'
schema = 'PUBLIC'
warehouse = 'adf_demo_wh'
role = 'sysadmin'

# Create connection object 
conn = snowflake.connector.connect(
    account=account,
    user=user,
    password=password,
    database=database,
    schema=schema,
    warehouse=warehouse,
    role=role
)


#####################################################################################################
# WEBSCRAPING FUNCTIONS
# ADD NEW WEBSCRAPING FUNCTION BELOW

def Azure_data_factory():
    import re
    response = requests.get("https://learn.microsoft.com/en-us/azure/data-factory/connector-overview")
    # Parse the HTML content of the webpage using Beautiful Soup
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find the table containing the data store names
    table = soup.find('table')
    table_body = table.find('tbody')
    rows = table_body.find_all('tr')
    data_stores = []
    for row in rows:
        cols = row.find_all('td')
        if cols[1].find('a'):
            data_stores.append(cols[1].find('a').string)
    
    final_list = []
    for i in data_stores:
        # Search for the pattern in the text using regex and return the match object
        final_list.append(i.replace(' (Preview)', ''))

    return final_list

def databricks():
    url = "https://docs.databricks.com/data/data-sources/index.html#data-sources"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    # Find all connector names within <span> tags with class "doc"
    connector_names = [span.text for span in soup.find_all("span", class_="doc")]
    return ['Python', 'Scala', 'Delta Lake', 'Delta Sharing', 'Parquet', 'ORC', 'JSON', 'CSV', 'Avro', 'JDBC', 'PostgreSQL', 
            'MySQL', 'MariaDB', 'SQL Server', 'Amazon Redshift', 
            'Google BigQuery', 'MongoDB', 'Cassandra', 'Couchbase', 'ElasticSearch', 'Snowflake', 'Azure Cosmos DB', 'Azure Synapse Analytics', 
            'LZO']

def synapse_pipeline():
   return Azure_data_factory()

def athena():
    import re
    url = "https://docs.aws.amazon.com/athena/latest/ug/connectors-prebuilt.html"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    connectors = soup.find_all('li')
    names = []
    for i in connectors:
         name = i.find_all('a')
         names.append(name)

    pattern = '>(.*)<'
    final_list = []
    for i in names:
        # Search for the pattern in the text using regex and return the match object
        match = re.findall(pattern, str(i))

        # If a match is found, print the matched pattern
        res = [*set(match)]
        if res != []:
            final_list.append(res[0])

    return final_list[3:]

def glue():
    url = "https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-etl-connect.html"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    connectors = soup.find_all('td')
    pattern = '>(.*)<'
    final_list = []
    for i in connectors:
        # Search for the pattern in the text using regex and return the match object
        #match = re.findall(pattern, str(i))
        final_list.append(i.text)
        # If a match is found, print the matched pattern
        #res = [*set(match)]
        #if res != []:
            #final_list.append(res[0])

    return ['Spark', 'Athena', 'JDBC', 'DocumentDB',  'DynamoDB', 'Kafka', 'Kinesis', 'MongoDB', 'MySQL', 'Oracle', 'Amazon S3',
             'Apache Hive', 'Parquet',  'PostgreSQL', 'Amazon Redshift', 'Microsoft SQL Server']

def emr():
    url = 'chatgpt'
    return ['Amazon S3', 'Amazon DynamoDB', 'Amazon Redshift', 'Amazon RDS', 
            'Apache Kafka', 'Apache HBase', 'Apache Hive', 'Apache Pig', 'Apache Hudi', 
            'Apache Cassandra', 'Elasticsearch', 'Google BigQuery', 'Microsoft SQL Server', 
            'MySQL', 'Oracle', 'PostgreSQL', 'Teradata', 'JDBC', 'Kinesis Data Firehose', 
            'Kinesis Data Streams', 'Splunk', 'Twitter', 'MongoDB']

def MWAA():
    url = 'chapgpt'
    return ['Amazon S3', 'Amazon Redshift', 'Amazon EMR', 'AWS Glue', 
            'Amazon SageMaker', 'Amazon DynamoDB', 'Amazon SQS', 'Amazon SNS', 
            'Amazon SimpleDB', 'Amazon EC2', 'Amazon ECS', 'AWS Batch', 
            'AWS Step Functions', 'Apache Cassandra', 'Apache Druid', 'Apache Hadoop', 
            'Apache Hive', 'Apache Pig', 'Apache Spark', 'FTP']

def cloudcomposer():
    url = "https://help.salesforce.com/s/articleView?id=sf.ms_composer_reference.htm&type=5"
    return ['Asana', 'Box', 'Gmail', 'Google Calendar', 'Google Sheets', 'HTTP', 'HubSpot', 
            'Jira', 'Marketo', 'Microsoft Dynamics 365 Business Central', 'Microsoft Teams', 
            'NetSuite', 'QuickBooks Online', 'MuleSoft RPA', 'Sage Intacct', 'Salesforce', 
            'Salesforce Marketing Cloud', 'ServiceNow', 'Slack', 'Snowflake', 'Stripe', 'Tableau', 
            'Twilio', 'Workday', 'Xero', 'Zendesk', 'Zuora']

def dataflow():
    import re
    url = 'https://learn.microsoft.com/en-us/connectors/connector-reference/'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    connectors = soup.find_all('b')
    #print(connectors)
    pattern = '>(.*)<'
    final_list = []
    for i in connectors:
        # Search for the pattern in the text using regex and return the match object
        match = re.findall(pattern, str(i))

        # If a match is found, print the matched pattern
        res = [*set(match)]
        if res != []:
            final_list.append(res[0].replace(' (Independent Publisher)', ''))

    return final_list

def matillion():
    #url = 'https://www.matillion.com/technology/integrations/#pricingTabs'
    #page = requests.get(url)
    #soup = BeautifulSoup(page.content, 'html.parser')
    #titles = [title.text for title in soup.find_all('span', class_='title')]
    titles = ['Amazon Aurora', 'Amazon Aurora (Unload)', 'Amazon DynamoDB', 
              'Amazon RDS', 'Amazon S3', 'Amazon S3 (Unload)', 'Amplitude', 'AOL Email', 
              'Apache Hive', 'Apache Spark', 'Azure Blob Storage', 
              'Azure Blob Storage (Unload)', 'Azure Cosmos DB', 'Azure SQL', 'Box', 
              'Dropbox', 'Dynamics', 'Dynamics 365 Sales', 'Dynamics CRM', 'Dynamics NAV', 
              'ElasticSearch', 'Email', 'Facebook', 'FTP', 'Gmail', 'Google Analytics', 
              'Google BigQuery', 'Google Cloud Storage', 'Google Custom Search', 
              'Google Sheets', 'HDFS', 'HTTP', 'HTTPS', 'HubSpot', 'IBM DB2', 
              'IBM DB2 for i', 'Instagram', 'Intercom', 'JIRA', 'LDAP', 'LinkedIn', 
              'LinkedIn Ads', 'Mailchimp', 'Mandrill', 'MariaDB', 'MariaDB (Unload)', 
              'Marketo', 'Microsoft Azure SQL (Output)', 'Microsoft Excel', 
              'Microsoft SharePoint', 'Microsoft SQL Server', 'MindSphere', 'Mixpanel', 
              'MongoDB', 'MySQL', 'MySQL (Unload)', 'Netezza', 'NetSuite', 'OData', 
              'Open Exchange Rates', 'Oracle', 'Oracle Eloqua', 'Outlook Email', 'Pardot', 
              'PayPal', 'PostGreSQL', 'PostgreSQL (Unload)', 'QuickBooks', 'Recurly', 
              'Redis', 'REST API', 'Sage Intacct', 'Salesforce', 'Salesforce (Output)', 
              'Salesforce Marketing Cloud', 'SAP', 'SAP Hana', 'SAP NetWeaver', 'SendGrid',
              'ServiceNow', 'SFTP', 'Shopify', 'Snapchat', 'Splunk', 'Square', 'Stripe', 
              'Sugar CRM', 'SurveyMonkey', 'Sybase ASE', 'Teradata', 'Twilio', 'Twitter', 
              'Twitter Ads', 'Webhook', 'Windows Fileshare', 'Xero', 'Yahoo Email', 
              'YouTube', 'YouTube Analytics', 'Zendesk', 'Zoho', 'Zoho Email']
    return titles

def talend():
    import re
    url = 'https://help.talend.com/r/en-US/Cloud/data-preparation-user-guide/list-of-supported-connectors'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    connectors = soup.find_all('td')
    #print(connectors)
    pattern = '>(.*)<'
    final_list = []
    for i in connectors:
        # Search for the pattern in the text using regex and return the match object
        match = re.findall(pattern, str(i))

        # If a match is found, print the matched pattern
        res = [*set(match)]
        if res != []:
            final_list.append(res[0])


    return ['Amazon Aurora', 'Amazon DynamoDB', 'Amazon Redshift', 'Apache Kudu', 
            'Azure Cosmos DB', 'Azure Synapse', 'Couchbase', 'Delta Lake', 'Derby', 
            'Google BigQuery', 'Google Bigtable', 'MariaDB', 'Microsoft SQL Server', 
            'MongoDB', 'MySQL', 'Oracle', 'PostgreSQL', 'SingleStore', 'Snowflake', 
            'Amazon S3', 'Azure Blob Storage', 'Azure Data Lake Storage Gen2', 'Box', 
            'Google Cloud Storage', 'Dynamics 365', 'Marketo', 'Google Analytics', 'NetSuite', 
            'Salesforce', 'Workday', 'Zendesk', 'REST', 'FTP', 'HDFS', 'Amazon Kinesis', 
            'Apache Pulsar', 'Azure Event Hubs', 'Google PubSub', 'Kafka', 'RabbitMQ', 
            'ElasticSearch']

def FiveTran():
    url = 'https://fivetran.com/connectors'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    print(soup)
    # find all connector names
    connectors = soup.find_all('div', {'class': 'card__json w-embed w-script'})
    names = []
    print(connectors)
    for connector in connectors:
        script = connector.find('script', {'type': 'application/json'})
        name = script.text.split('\n')[2][8:].replace(',','')
        names.append(name.replace('"', ''))
    
    file = open("fivetran.txt")
    # read the file as a list
    data = file.readlines()
    # close the file
    file.close()
    final = []
    count = 0
    for i in data:
        if (len(i) < 40) and (i != '\n') and (i != 'Lite\n'):
            final.append(i)
        count+=1

    res = [sub[: -1] for sub in final]

    return res[0:len(res) - 1]
    

def Airflow():
    url = "https://airflow.apache.org/docs/apache-airflow-providers/core-extensions/connections.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    connector_table = soup.find_all('ul')[70]
    names = []
    for a in connector_table.find_all('a'):
        names.append(a.text)
    return names

def prefect():
    return FiveTran()

def dagster():
    reference = 'https://help.talend.com/r/en-US/Cloud/data-preparation-user-guide/list-of-supported-connectors'
    titles = ["Airbyte","Airflow","AWS","Azure","Celery ","Docker","Kubernetes","Census","Dask",
    "Databricks","Datadog","Datahub","DuckDB","Fivetran","GCP",
    "Great Expectations","GitHub","GraphQL","MLflow","Microsoft Teams","MySQL","PagerDuty",
    "Pandas","Papertrail","PostgreSQL","Prometheus","Pyspark","Shell","Slack","Snowflake",
    "Spark","Twilio"]
    return titles

def dbt():
    import re
    url = 'https://docs.getdbt.com/docs/supported-data-platforms'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    connectors = soup.find_all('td')
    names = []
    for i in connectors:
        names.append(i.text)
    return ['AlloyDB', 'Azure Synapse', 'BigQuery','Databricks',  'Dremio', 'Postgres',  'Redshift', 'Snowflake', 'Spark', 'Starburst & Trino',  'Athena',
             'Greenplum', 'Oracle', 'Clickhouse', 'Hive', 'Rockset', 'IBM DB2', 'Impala', 'SingleStore', 'Doris & SelectDB', 'Infer', 'SQLite', 'DuckDB', 
             'iomete', 'SQL Server',  'Layer', 'Teradata', 'Exasol Analytics', 'Materialize', 'TiDB', 'Firebolt', 'MindsDB', 'Vertica', 'AWS Glue', 'MySQL', 
               'Databend Cloud']

def airbyte():
    url = 'https://airbyte.com/connectors?a1b8d32e_page=2'
    # open the data file
    file = open("airbyte.txt")
    # read the file as a list
    data = file.readlines()
    # close the file
    file.close()
    final = []
    count = 0
    for i in data:
        if count % 4 == 0:
            final.append(i)
        count+=1

    res = [sub[: -1] for sub in final]

    return res

def stitch():
    url = 'https://www.stitchdata.com/integrations/sources/'
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    connectors = soup.find_all('p')
    final = []
    for i in connectors: 
            final.append(i.text)
    count = 0
    res = []
    for i in final[5:]:
        if count % 2 == 0:
            res.append(i)
        count+=1
    return res[:-4]

#####################################################################################################
# INSTRUCTIONS

#1) ADD INTEGRATION TOOL TO INTEGRATIONTOOL TABLE
#2) MANUALLY FIND SECURITY AND SUPPORT FEATURES FOR THAT TOOL (chatgpt or tool website)
#3) IF THERE IS A FEATURE THAT ISNT PRESENT IN THE SECURITY/SUPPORT FEATURES TABLES, THEN ADD IT
#4) MANUALLY MAP SECURITY AND SUPPORT FEATURES TO THE NEW INTEGRATION TOOL
#5) CREATE WEBSCRAPING SCRIPT TO EXTRACT ALL CONNECTORS FROM THE INTEGRATION TOOL WEBSITE
#6) MAKE SURE THE FUNCTION RETURNS A LIST OF STRINGS
#7) FEED NEWLY MADE FUNCTION OUTPUT INTO add_new_connectors_and_map(lst, integrationid)
#8) EXAMPLE: add_new_connectors_and_map(Azure_Data_Factory(), 6)
#9) THE TABLES AND REPORTS ARE NOW UPDATED WITH AN ADDITIONAL INTEGRATION TOOL
######################################################################################################

def add_new_connectors_and_map(lst, integrationid):
    # Create a cursor to execute queries
    cur = conn.cursor()
    for i in lst: 
    # Execute the procedure with the input parameter
        cur.callproc('add_connector_if_not_exists', [i])
    # Commit the transaction
        conn.commit()

    # Close the cursor and connection
    cur.close()
    

    cursor = conn.cursor()

    for i in lst:
        # Query the connectors table for the connectorid corresponding to the connectorname
        cursor = conn.cursor()
        query = f"SELECT connectorid FROM connector WHERE UPPER(connectorname) = UPPER('{i}')"
        cursor.execute(query)
        result = cursor.fetchone()
        connectorid = result[0]
        query = f"INSERT INTO IntegrationTool_Connector_Map (integrationtoolID, connectorid) VALUES ('{integrationid}', '{connectorid}')"
        cursor.execute(query)
        conn.commit()

    conn.close()

#7) FEED NEWLY MADE FUNCTION OUTPUT INTO add_new_connectors_and_map(lst, integrationid)

#add_new_connectors_and_map(stitch(), 26)

print(athena())


