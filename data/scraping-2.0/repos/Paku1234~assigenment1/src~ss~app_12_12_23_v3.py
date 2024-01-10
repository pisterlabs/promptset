from flask import Flask, jsonify, request, render_template, send_from_directory
import os
from flask_cors import CORS
import pandas as pd
import numpy as np
import openai


app = Flask(__name__, template_folder='D:/multi-page-app/src/backend/templates')
CORS(app)  # Enable CORS for all routes

# Read the Assignment_Data file
df = pd.read_csv("D://multi-page-app//src//data//Assignment_Data.csv")

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
# Extract the year and quarter
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter

# OPEN_AI_KEY = os.getenv('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')

# openai.api_key = OPEN_AI_KEY

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/sales-page')
def sales_page():
    return render_template('sales.html')

# Routes for serving static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Helper function to get unique values for a column
def get_unique_values(column_name):
    return df[column_name].unique().tolist()

@app.route('/count', methods=['GET'])
def get_counts():
    # Get total unique stores and departments
    total_Store = get_unique_values('Store')
    total_Department = get_unique_values('Department')

    data = {
        'total_Store': total_Store,
        'total_Department': total_Department
    }
    return jsonify(data)

@app.route('/filter', methods=['POST'])
def filter_data():
    data = request.get_json()
    data_date = pd.to_datetime(data['date'], format="%Y-%m-%d")
    # Filter data based on selected store and department
    filtered_df = df[(df['Store'] == np.int64(data['store'])) & (df['Date'] == data_date) & (df['Department'] == np.int64(data['department']))]

    print(filtered_df)

    # Convert filtered data to dictionary format
    result = filtered_df.to_dict('records')
    print("Results", result)

    return jsonify(result)

@app.route('/holiday', methods=['GET', 'POST'])
def check_holiday():
    data = request.get_json()
    store = data['Store']
    date = data['Date']
    department = data['epartment']

    # Check if the query result is empty before accessing elements
    query_result = df[(df['Store'] == store) & (df['Date'] == date) & (df['Department'] == department)]['IsHoliday'].values
    print('query_result:',query_result)

    if len(query_result) > 0:
        is_holiday = query_result[0]
        result = {'is_holiday': is_holiday}
    else:
        result = {'is_holiday': None}

    return jsonify(result)

@app.route('/sales_data', methods=['POST'])
def get_sales_data():
    data = request.get_json()
    print("Data", data)
    data_date = np.int32(data['year'])
    # Filter data based on selected store, departments, year, and quarter
    filtered_df = df[(df['Store'] == np.int64(data['store'])) & 
                  (df['Department'].isin(np.int64(data['departments']))) & 
                  (df['Year'] == data_date) & 
                  (df['Quarter'] == np.int32(data['quarter']))]

    print("filtered_df",filtered_df)
    # Convert filtered data to dictionary format
    result = filtered_df.to_dict('records')
    print("Sales_dataResults",result)

    return jsonify(result)

@app.route('/modified_sales_data', methods=['POST'])
def get_modified_sales_data():
    data = request.get_json()
    print("Data", data)
    # Filter data based on selected store, department, and date
    filtered_df = df[(df['Store'] == np.int64(data['store'])) & 
                  (df['Department'] == np.int64(data['department']))]

    print("filtered_df",filtered_df)
    # Convert filtered data to dictionary format
    result = filtered_df.to_dict('records')
    print("Sales_dataResults",result)

    return jsonify(result)


@app.route('/chatbot', methods=['POST'])
def chatbot():
   message = request.json['message']
   response = openai.Completion.create(
       engine="text-davinci-002",
       prompt=message,
       max_tokens=150
   )
   return jsonify(response.choices[0].text.strip())

if __name__ == '__main__':
    app.run(debug=True, port= 5000)

