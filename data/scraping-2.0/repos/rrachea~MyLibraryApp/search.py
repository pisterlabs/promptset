# import useful libraries
from flask import Flask, jsonify, request
import pandas as pd
import openai
openai.api_key = "xxxx"
import pandas_gpt

###### Useful Functions ######
"""
Input array:
Object = {
    'method' : 'check_db',
    'context' : 'find a book that contains ship',
    'student_id' : '5d66ee1de283ad380812600cbc8859789df54b15425ac883a5ff32a941248115',
    'call_number' : 'JA66 .H795 1968',
    'name' : 'The Obama phenomenon'
}
"""

# load datasets
physical_df = pd.read_excel('./datasets/Physical.xlsx')
online_df = pd.read_excel('./datasets/Online.xlsx')
glo_df = pd.read_excel('./datasets/GLO.xlsx')
# print("Datasets loaded")
# print(glo_df.head())

app = Flask(__name__)
@app.route('/api/data', methods=['GET'])
def search_db(obj):
    context = obj['context']
    # check glo
    if 'graduate learning outcome' in context or 'glo' in context:
        # perform check
        student_details = glo_df.ask(f"find the record with {obj['student_id']} as a dictionary with the column name as the key and the record data as the value. Remove the StudentID2 key and all keys with nan values.")
        # something, return first details
        if len(student_details) > 0:
            return jsonify(student_details)
        else:
            return None

    # check physical and online items
    else:
        # search physical
        try:
            resource_details = physical_df.ask(f"find records with {obj['call_number']} and store it as a dictionary with the column name as the key and the record data as the value. Replace all nan values with 'Not available'. Store datetime values as string with the year, month and date.")
            return jsonify(resource_details)
        # search online
        except:
            resource_details = online_df.ask(f"find records that contain {obj['name']}. Store the column names as keys and the record data as value as a dictionary.")
            return jsonify(resource_details)
        finally:
            return None

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=5000)