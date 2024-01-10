# import necessary libraries
import boto3
from trp import Document
import pandas as pd
# import OpenAI Library
import openai
import json
import csv
# Authenticate with your OpenAI API Key
openai.api_key = "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"

# Return Bytes From Wasabi S3
def get_object(access_key, secret_key, region, endpoint, bucket, file):

    # Create a Wasabi S3 client object
    s3 = boto3.client('s3',
                      endpoint_url=endpoint,
                      aws_access_key_id=access_key,
                      aws_secret_access_key=secret_key,
                      region_name=region)

    bucket_name = bucket
    key_name = file

    file_object = s3.get_object(Bucket=bucket_name, Key=key_name)
    file_content = file_object['Body'].read()
    return file_content

# Create Textract Service
def create_textract(region, access_key, secret_key):
    textract = boto3.client('textract',
                            region_name=region,
                            aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key
                            )
    return textract

# Analysis Invoice Document of S3 Bucket
def analyze_invoice(textract, file):
    response = textract.analyze_expense(
        Document={
            'Bytes' : file
            # 'S3Object': {
            #     'Bucket': s3,
            #     'Name': file
            # }
        })
    return response

# Get Document Type
def type_invoice(textract, file):
    response = textract.detect_document_text(
        Document={
            'Bytes' : file
            # 'S3Object': {
            #     'Bucket': s3,
            #     'Name': file
            # }
        })
    toLower = json.dumps(response).lower()
    index = toLower.find('"text": "purchase order"')
    if index > 0:
        return "PURCHASE_ORDER"

    index = toLower.find('"text": "packing slip"')
    if index > 0:
        return "PACKING_SLIP"

    index = toLower.find('"text": "receiving slip"')
    if index > 0:
        return "RECEIVING_SLIP"
    
    index = toLower.find('"text": "credit"')
    if index > 0:
        return "CREDIT_MEMO"

    index = toLower.find('"text": "invoice"')
    if index > 0:
        return "INVOICE"

    index = toLower.find('"text": "quote"')
    if index > 0:
        return "QUOTE"

    return "OTHER"

# Summary Analyze
def get_summary(response, filepath):
    summaryDocument = {
        document["ExpenseIndex"]: document["SummaryFields"]
        for document in response["ExpenseDocuments"]
    }
    summary = []
    # Get Summary Field
    for item in summaryDocument:
        for item_summary in summaryDocument[item]:
            temp_summary = {}
            if "Type" in item_summary:
                temp_summary["Type"] = item_summary["Type"]["Text"]
            else:
                temp_summary["Type"] = "NONE"

            if "ValueDetection" in item_summary:
                temp_summary["Value"] = item_summary["ValueDetection"]["Text"]
            else:
                temp_summary["Value"] = "NONE"

            if "GroupProperties" in item_summary:
                temp_summary["Group"] = item_summary["GroupProperties"][0]["Types"][0]
            else:
                temp_summary["Group"] = "NONE"

            if "LabelDetection" in item_summary:
                temp_summary["Label"] = item_summary["LabelDetection"]["Text"]
            else:
                temp_summary["Label"] = "NONE"

            summary.append(temp_summary)
    # Save JSON of Summary Into CSV File
    data_file = open('./CSV/index-{}.csv'.format(filepath), 'w')
    csv_writer = csv.writer(data_file)
    count = 0
    for item in summary:
        if count == 0:
            header = ["Type", "Value", "Group", "Label"]
            csv_writer.writerow(header)
            count += 1
        csv_writer.writerow(item.values())
    data_file.close()

# Table Analyze
def get_table(response, filepath):
    # Get Data From Returned JSON From Textract
    tableDocument = {
        document["ExpenseIndex"]: document["LineItemGroups"]
        for document in response["ExpenseDocuments"]
    }
    # Get Table Field
    table = []
    for item in tableDocument:
        for table_item in tableDocument[item]:
            for table_line_item in table_item["LineItems"]:
                table.append(table_line_item["LineItemExpenseFields"])

    table_field = []
    for row in table:
        temp_table_row = []
        for col in row:
            temp_table_col = {}
            if "Type" in col:
                temp_table_col["Type"] = col["Type"]["Text"]
            else:
                temp_table_col["Type"] = "NONE"

            if "ValueDetection" in col:
                temp_table_col["Value"] = col["ValueDetection"]["Text"]
            else:
                temp_table_col["Value"] = "NONE"

            if "LabelDetection" in col:
                temp_table_col["Label"] = col["LabelDetection"]["Text"]
            else:
                temp_table_col["Label"] = "NONE"
            temp_table_row.append(temp_table_col)
        table_field.append(temp_table_row)

    table_result = [] 
    for item in table_field:
        table_obj = {}
        for col in item:
            value = col["Value"].replace("$", "").replace("\n", " ")
            try:
                float(value)
            except:
                table_obj[col["Type"]] = value
            else:
                table_obj[col["Type"]] = float(value)

        table_result.append(table_obj)

    return table_result
    
