import openai
import csv
import re
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
from openai import openai
#Data creation
def data_create(input_data):
    data_generation  = openai("create a table that contain 10 names")
    # Split the table content into rows
    rows = [row.split('|')[1:-1] for row in data_generation.strip().split('\n')]
    # Remove leading/trailing spaces from each cell
    rows = [[cell.strip() for cell in row] for row in rows]
    # Extract the header row
    header = rows[0]
    # Extract the data rows
    data = rows[1:]
    

    # Write the table content to the CSV file
    with open(input_data, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header
        writer.writerows(data)   # Write the data rows
    print(f"Table content saved as CSV file")

#Script creation
def script(jmx_file ):
    Openai_Response  = openai('''Hello, I need an XML file for a JMeter version 5.5 test plan to 
                              perform a POST request on the following API: https://reqres.in/api/users API boday data { "name": "morpheus", "job": "leader" }
                              and also add summary report listener to the thread group''')
    #Openai_Response  = openai(input)
    with open(jmx_file, "w", encoding="utf-8") as file:
        lines = Openai_Response.splitlines()
        # Remove the first and last lines
        trimmed_lines = lines[1:-1]
       
        # Combine the remaining lines back into a single string
        modified_xml = "\n".join(trimmed_lines)
        # Remove the '''xml in the first line
        modified_xml = re.sub(r"'''xml", "", modified_xml, count=1).strip()
        # Remove the ''' in the last line
        modified_xml = re.sub(r"'''", "", modified_xml, count=1).strip()
        file.write(modified_xml)
        return "JMeter Script created created sucessfully"

#Running Jmeter Script -------------------------------------------------------------------------------
def execute_jmeter_script(jmeter_path, jmx_file_path, log_file_path):
    """
    Execute a JMeter script using the command-line interface (CLI).
    
    Parameters:
        jmeter_path (str): The path to the JMeter executable (e.g., '/path/to/jmeter/bin/jmeter').
        jmx_file_path (str): The path to the JMeter test plan (JMX file) to execute.
        log_file_path (str): The path to the log file where JMeter output will be saved.
    """
    # Construct the command to run JMeter
    command = [
        jmeter_path,
        '-n',  # Non-GUI mode
        '-t', jmx_file_path,  # Path to the JMX file
        '-l', log_file_path,  # Path to the log file
    ]

    try:
        # Execute the JMeter script
        subprocess.run(command, check=True)
        print("JMeter script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"JMeter script execution failed with error: {e}")

#Get data and creating XML file-------------------------------------------------------------------------------------------
def extract_metrics_from_csv(csv_file_path, output_file_path):
    """
    Extracts metrics from the CSV file and saves them in an XML file.

    Parameters:
        csv_file_path (str): The path to the CSV file (e.g., '/path/to/csv_file.csv').
        output_file_path (str): The path to save the XML output (e.g., '/path/to/output.xml').
    """
    print(csv_file_path)
    print(output_file_path)
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file_path)

    # Calculate metrics
    grouped_df = df.groupby('label').agg({
        'elapsed': ['mean', 'count'],
        'success': lambda x: (x == True).sum(),
    })

    # Compute the error column
    grouped_df['error'] = (1 - grouped_df[('success', '<lambda>')]/grouped_df[('elapsed', 'count')]) * 100

    # Drop unnecessary columns from the DataFrame
    grouped_df = grouped_df[['elapsed', 'error']]

    # Rename columns to desired format
    grouped_df.columns = ['mean', 'count', 'error']

    # Calculate additional metrics
    grouped_df['median'] = df.groupby('label')['elapsed'].median()
    grouped_df['average'] = df.groupby('label')['elapsed'].mean()
    grouped_df['maximum'] = df.groupby('label')['elapsed'].max()
    grouped_df['minimum'] = df.groupby('label')['elapsed'].min()
    grouped_df['percentile90'] = df.groupby('label')['elapsed'].quantile(0.90)
    grouped_df['percentile95'] = df.groupby('label')['elapsed'].quantile(0.95)
    grouped_df['percentile99'] = df.groupby('label')['elapsed'].quantile(0.99)

    # Reset index to convert 'label' from the index to a regular column
    grouped_df.reset_index(inplace=True)

    # Rename the 'label' column to 'name'
    grouped_df.rename(columns={'label': 'name'}, inplace=True)

    # Save the metrics to an XML file
    root = ET.Element('metrics')
    for _, row in grouped_df.iterrows():
        item = ET.SubElement(root, 'item', name=row['name'])
        for metric_name, metric_value in row.items():
            ET.SubElement(item, metric_name).text = str(metric_value)

    # Serialize the XML data to a string
    xml_data = ET.tostring(root, encoding='unicode')

    tree = ET.ElementTree(root)
    tree.write(output_file_path)
    print(xml_data)
    Openai_Response  = openai(f"{xml_data}+Assume yourself as Performance architect and give you insights fo create a repot with data provided")
    return Openai_Response

#Report part-------------------------------------------------------------------------------------------------
def convert_xml_to_html(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    table_html = "<table id='myTable'>"
    table_html += "<thead><tr>"

    headers = []
    for child in root[0]:
        headers.append(child.tag)
        table_html += "<th>{}</th>".format(child.tag)
    table_html += "</tr></thead><tbody>\n"

    for element in root:
        table_html += "<tr>"
        for child in element:
            table_html += "<td>{}</td>".format(child.text)
        table_html += "</tr>\n"

    table_html += "</tbody></table>"
    return table_html

#Create Report page------------------------------------------------------------------------------------------
def generate_index_html(html_data , xml , existing_html ):

    with open(f"{existing_html}", "r") as file:
        template_html = file.read()

    modified_html = template_html.replace('<p class="table-placeholder">Table will be inserted here</p>', html_data)
    modified_html_1 = modified_html.replace('<p class="openai-response">Table will be inserted here</p>',xml)
    
    with open("index.html", 'w') as file:
        file.write(modified_html_1)
        print("Report created")

def analyze_script(file_path):
    
    try:
        # Step 1: Open the file in read mode
        with open(file_path, 'r') as file:
            # Step 2: Read the contents of the file and store them in a variable
            file_contents = file.read()

        # Step 3: File automatically closes once the 'with' block is exited
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except IOError:
        print(f"Error reading the file: {file_path}")

    Openai_Response  = openai(f"{file_contents}+Assume yourself as a performance enginner and find paramterization ,corelations in this jmeter script")
    return Openai_Response

def security():
    
    with open("Code\snake.py", "r") as source_file:
        source_content = source_file.read()

    Openai_Response  = openai(f"{source_content}+Assume yourself as a security architect and analysis code Security give report")
    return Openai_Response