from fredapi import Fred
import pandas as pd
import pandas as pd
import numpy as np
from scipy.stats import norm
import numpy as np
import openai
import os
from dotenv import load_dotenv
import json
import csv
from datetime import datetime, timedelta
from mysqlconnection import connectToMySQL
import jsonlines

# Load the environment variables from the .env file
load_dotenv()

fred_api_key = os.getenv("FRED_API_KEY")
openai.api_key = os.getenv('OPENAI_API_KEY')

@classmethod
def create_guage_png(cls):

    fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = 270,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Speed"}))
    
    # Set the background color to black
    fig.update_layout(
        plot_bgcolor='black'
    )
    
    fig.show()
    # fig.write_image("Final_Project/flask_app/models/risk_meter.png")

@classmethod
def create_guage_img_str(cls):
    
    # Create the gauge figure
    gauge_fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = 6.22,
    gauge = {
        'axis': {'range': [0, 7], 'tickwidth': 1, 'tickcolor': 'darkblue'},
        'bar': {'color': 'red', 'thickness': 0.3},
        'steps': [
            {'range': [0, 1], 'color': 'green'},
            {'range': [1, 2], 'color': 'yellowgreen'},
            {'range': [2, 3], 'color': 'yellow'},
            {'range': [3, 4], 'color': 'gold'},
            {'range': [4, 5], 'color': 'orange'},
            {'range': [5, 6], 'color': 'darkorange'},
            {'range': [6, 7], 'color': 'red'}
            ]
        },
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))

    # # Convert the figure to a PNG image
    # img_bytes = gauge_fig.to_image(format='png')

    # # Encode the PNG image as a Base64 string
    # img_str = base64.b64encode(img_bytes).decode()
    
    # Set the background color to black
    gauge_fig.update_layout(
        plot_bgcolor='black'
    )

    gauge_fig.show()
    # img = gauge_fig.write_image("Final_Project/flask_app/models/risk_meter_2.png")

    @classmethod
    def create_chart(cls):
        
        # data
        df = px.data.gapminder().query("continent=='Oceania'")

        # plotly express bar chart
        fig = px.line(df, x="year", y="lifeExp", color='country')

        # html file
        # plotly.offline.plot(fig, filename='Final_Project/flask_app/models/lifeExp.html')

# Create Empty DataFrame
def create_empty_df(df):
    
    # Create a DataFrame with 0 in every value and 469 rows and 9 columns
    results_df = pd.DataFrame(0, index=range(len(df)), columns=df.columns)

    # Print the first 5 rows of the DataFrame to check that it was created correctly
    # print(results_df.head())
    
    return results_df

# Generate Date From FRED API 
def get_data(api_key):
    
    # Replace YOUR_API_KEY with your actual API key
    fred = Fred(api_key=api_key) 

    # Define the list of series IDs to download
    # series_ids = ['CPILFESL', 'PCECTrimMED', 'CPILFESLNO', 'CPIMEDSL', 'CPILFSSM', 'CPILFSEXA', 'CPILFVOTT01', 'CPILFV01', 'CPITRIM1M162N']
    series_ids = ['STICKCPIM159SFRBATL',
                'STICKCPIXSHLTRM159SFRBATL',
                'CORESTICKM159SFRBATL',
                'CRESTKCPIXSLTRM159SFRBATL',
                'PCETRIM12M159SFRBDAL',
                'TRMMEANCPIM159SFRBCLE',
                'MEDCPIM159SFRBCLE',
                'FLEXCPIM159SFRBATL',
                'COREFLEXCPIM159SFRBATL']

    # Download the data for each series and store it in a dictionary
    data = {}
    for series_id in series_ids:
        series_data = fred.get_series(series_id)
        data[series_id] = series_data

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data)
    print(df)

    # # Reset the index to be the date
    df = df.reset_index()

    # Rename the columns to be more descriptive
    df = df.rename(columns={'index': 'date',
                            'STICKCPIM159SFRBATL': 'Sticky_Price_CPI',
                            'STICKCPIXSHLTRM159SFRBATL': 'Sticky_Price_CPI_Less_Shelter',
                            'CORESTICKM159SFRBATL': 'Sticky_Price_CPI_Less_Food_Energy',
                            'CRESTKCPIXSLTRM159SFRBATL': 'Sticky_Price_CPI_Less_Food_Energy_Shelter',
                            'PCETRIM12M159SFRBDAL': 'Trimmed_Mean_PCE_Inflation_Rate',
                            'TRMMEANCPIM159SFRBCLE': '16_Percent_Trimmed_Mean_CPI',
                            'MEDCPIM159SFRBCLE': 'Median_CPI',
                            'FLEXCPIM159SFRBATL': 'Flexible_Price_CPI',
                            'COREFLEXCPIM159SFRBATL': 'Flexible_Price_CPI_Less_Food_Energy'})

    # Convert the date column to datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Set the date column as the index
    df = df.set_index('date')
    print(df)

    # Resample the data to be monthly and calculate the monthly percentage change
    #df = df.resample('M').last().pct_change()

    # Drop the first row, which will have NaN values due to the percentage change calculation
    df = df.drop(df.index[0])
    print(df)

    # Display the resulting DataFrame
    # print(df.describe())
    # print(df.head())
    
    # Drop columns with missing data
    df_filtered = df.dropna()
    print(df_filtered) 
    
    # df.to_excel("output.xlsx") 
    df_filtered.to_csv('Python_Wks_3-7/Final_Project/flask_app/models/data/data.csv') 
    
# get_data(fred_api_key)

# Generate Last Date Only From FRED API
def get_last_date(api_key):
    
    # Replace YOUR_API_KEY with your actual API key
    fred = Fred(api_key=api_key) 

    # Define the list of series IDs to download
    # series_ids = ['CPILFESL', 'PCECTrimMED', 'CPILFESLNO', 'CPIMEDSL', 'CPILFSSM', 'CPILFSEXA', 'CPILFVOTT01', 'CPILFV01', 'CPITRIM1M162N']
    series_ids = ['STICKCPIM159SFRBATL',
                'STICKCPIXSHLTRM159SFRBATL',
                'CORESTICKM159SFRBATL',
                'CRESTKCPIXSLTRM159SFRBATL',
                'PCETRIM12M159SFRBDAL',
                'TRMMEANCPIM159SFRBCLE',
                'MEDCPIM159SFRBCLE',
                'FLEXCPIM159SFRBATL',
                'COREFLEXCPIM159SFRBATL']

    # calculate the start and end dates
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=91)).strftime('%Y-%m-%d')
    print(end_date)
    print(start_date)


    # Download the data for each series and store it in a dictionary
    data = {}
    for series_id in series_ids:
        # get the most recent observation for the series
        series_data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        data[series_id] = series_data
    # print(data)

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(data)
    print(df)

    # Reset the index to be the date
    df = df.reset_index()

    # Rename the columns to be more descriptive
    df = df.rename(columns={'index': 'Date',
                            'STICKCPIM159SFRBATL': 'Sticky_Price_CPI',
                            'STICKCPIXSHLTRM159SFRBATL': 'Sticky_Price_CPI_Less_Shelter',
                            'CORESTICKM159SFRBATL': 'Sticky_Price_CPI_Less_Food_Energy',
                            'CRESTKCPIXSLTRM159SFRBATL': 'Sticky_Price_CPI_Less_Food_Energy_Shelter',
                            'PCETRIM12M159SFRBDAL': 'Trimmed_Mean_PCE_Inflation_Rate',
                            'TRMMEANCPIM159SFRBCLE': '16_Percent_Trimmed_Mean_CPI',
                            'MEDCPIM159SFRBCLE': 'Median_CPI',
                            'FLEXCPIM159SFRBATL': 'Flexible_Price_CPI',
                            'COREFLEXCPIM159SFRBATL': 'Flexible_Price_CPI_Less_Food_Energy'})

    # Convert the date column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Set the date column as the index
    df = df.set_index('Date')

    # # Resample the data to be monthly and calculate the monthly percentage change
    # df = df.resample('M').last().pct_change()

    # Drop the first row, which will have NaN values due to the percentage change calculation
    # df = df.drop(df.index[0])

    # Display the resulting DataFrame
    print(df.head())
    
    # df.to_excel("output.xlsx") 
    df.to_csv(r'recent_data.csv') 
# get_last_date(fred_api_key)

# Risk Analysis 
def risk_analysis():
    
    # Import CSV from file folder.
    currentData = pd.read_csv('Python_Wks_3-7/Final_Project/flask_app/models/data/data.csv')
    print(currentData)
    
    # Splice out the date column and other non necessary columns
    dateColumn = currentData.iloc[:,:1].copy()
    print(dateColumn)
    
    # Remaining data minus date column
    df = currentData.iloc[:,1:].copy()
    print(df)

    results_df = create_empty_df(df)
    print(results_df)
    
    percentiles = [0,.125,.25,.375,.5,.625,.75,.875,1]
    
    for column in range(len(df.columns)):
        
        # categorize each value in one column based on risk level bawed on normal curve percentiles
        for row in range(len(df.iloc[:,column])):
            if df.iloc[:,column][row] >= df.iloc[:,column].describe(percentiles).loc['0%'] and df.iloc[:,column][row] < df.iloc[:,column].describe(percentiles).loc['12.5%']:
                # print('This is risk level 0')
                results_df.iloc[:,column][row] = 0
            elif df.iloc[:,column][row] >= df.iloc[:,column].describe(percentiles).loc['12.5%'] and df.iloc[:,column][row] < df.iloc[:,column].describe(percentiles).loc['25%']:
                # print('This is risk level 1')
                results_df.iloc[:,column][row] = 1
            elif df.iloc[:,column][row] >= df.iloc[:,column].describe(percentiles).loc['25%'] and df.iloc[:,column][row] < df.iloc[:,column].describe(percentiles).loc['37.5%']:
                # print('This is risk level 2')
                results_df.iloc[:,column][row] = 2
            elif df.iloc[:,column][row] >= df.iloc[:,column].describe(percentiles).loc['37.5%'] and df.iloc[:,column][row] < df.iloc[:,column].describe(percentiles).loc['50%']:
                # print('This is risk level 3')
                results_df.iloc[:,column][row] = 3
            elif df.iloc[:,column][row] >= df.iloc[:,column].describe(percentiles).loc['50%'] and df.iloc[:,column][row] < df.iloc[:,column].describe(percentiles).loc['62.5%']:
                # print('This is risk level 4')
                results_df.iloc[:,column][row] = 4
            elif df.iloc[:,column][row] >= df.iloc[:,column].describe(percentiles).loc['62.5%'] and df.iloc[:,column][row] < df.iloc[:,column].describe(percentiles).loc['75%']:
                # print('This is risk level 5')
                results_df.iloc[:,column][row] = 5
            elif df.iloc[:,column][row] >= df.iloc[:,column].describe(percentiles).loc['75%'] and df.iloc[:,column][row] < df.iloc[:,column].describe(percentiles).loc['87.5%']:
                # print('This is risk level 6')
                results_df.iloc[:,column][row] = 6
            elif df.iloc[:,column][row] >= df.iloc[:,column].describe(percentiles).loc['87.5%'] and df.iloc[:,column][row] <= df.iloc[:,column].describe(percentiles).loc['100%']:
                # print('This is risk level 7')
                results_df.iloc[:,column][row] = 7
            else:
                print('Error')
    
    # Add a column of the sum of each row
    results_df['mean'] = np.round(results_df.mean(axis=1), 4)

    # Print the updated DataFrame
    print(results_df)
    
    # Add back the dateColumn
    results_df.insert(0,'date', dateColumn)
    print(results_df)
    
    results_df.to_csv('Python_Wks_3-7/Final_Project/flask_app/models/data/recent_results.csv')
    
# risk_analysis()

# Save historical data into database
# Make sure to create a user before running this.
def save_econ_data():
    
    econ_data = pd.read_csv(r'Final_Project/flask_app/models/combined_data.csv')
    econ_data = econ_data.drop(['Unnamed: 0'], axis=1)
    
    # # Create User 1 so the data can be uploaded
    # user_1_query = 'INSERT INTO users (first_name, last_name, email, password) VALUES ("Nicolas", "Joye", "nicolasajoye@gmail.com", "12345");'
    # connectToMySQL('risk_analysis').query_db(user_1_query)
    
    for x in range(len(econ_data)):
    
        # print(econ_data.iloc[x])
        
        # Convert the string to a datetime object
        datetime_obj = datetime.strptime(econ_data['Date'][x], '%m/%d/%Y')

        # Format the datetime object as a string in the desired format
        formatted_date = datetime_obj.strftime('%Y-%m-%d')

        data = {
            'date': formatted_date,
            'spcpi': econ_data['Sticky_Price_CPI'][x],
            'spcpi_m_s': econ_data['Sticky_Price_CPI_Less_Shelter'][x],
            'spcpi_m_fe': econ_data['Sticky_Price_CPI_Less_Food_Energy'][x],
            'spcpi_m_fes': econ_data['Sticky_Price_CPI_Less_Food_Energy_Shelter'][x],
            'trim_mean_pce': econ_data['Trimmed_Mean_PCE_Inflation_Rate'][x],
            'sixteenP_trim_mean_cpi': econ_data['16_Percent_Trimmed_Mean_CPI'][x],
            'median_cpi': econ_data['Median_CPI'][x],
            'fpcpi': econ_data['Flexible_Price_CPI'][x],
            'fpcpi_m_fe': econ_data['Flexible_Price_CPI_Less_Food_Energy'][x],
        }
        # print(data)
        
        query = """INSERT INTO econ_data (date , spcpi , spcpi_m_s, spcpi_m_fe , spcpi_m_fes , trim_mean_pce, sixteenP_trim_mean_cpi, median_cpi, fpcpi, fpcpi_m_fe, created_at, updated_at, user_id)
                VALUES (%(date)s , %(spcpi)s , %(spcpi_m_s)s, %(spcpi_m_fe)s , %(spcpi_m_fes)s , %(trim_mean_pce)s , %(sixteenP_trim_mean_cpi)s, %(median_cpi)s , %(fpcpi)s , %(fpcpi_m_fe)s , NOW(), NOW(), 1);"""
        # print(query)
                
        # id = return connectToMySQL('risk_analysis').query_db(query, data)
        connectToMySQL('python_final_project').query_db(query, data)
# save_econ_data()

# Save risk_analysis data into database
# Make sure to create a user before running this.
# Does not work- Getting this error and the file is present FileNotFoundError: [Errno 2] No such file or directory: 'recent_results.csv'
def save_risk_data():
    
    risk_data = pd.read_csv('Python_Wks_3-7/Final_Project/flask_app/models/data/recent_results.csv')
    risk_data = risk_data.drop(['Unnamed: 0'], axis=1)
    print(risk_data)
    
    for x in range(len(risk_data)):
    
        # print(risk_data.iloc[x])
        
        # Convert the string to a datetime object
        # datetime_obj = datetime.strptime(risk_data['date'][x], '%m/%d/%Y')
        datetime_obj = datetime.strptime(risk_data['date'][x], '%Y-%m-%d')

        # Format the datetime object as a string in the desired format
        formatted_date = datetime_obj.strftime('%Y-%m-%d')

        data = {
            'date': formatted_date,
            'spcpi': risk_data['Sticky_Price_CPI'][x],
            'spcpi_m_s': risk_data['Sticky_Price_CPI_Less_Shelter'][x],
            'spcpi_m_fe': risk_data['Sticky_Price_CPI_Less_Food_Energy'][x],
            'spcpi_m_fes': risk_data['Sticky_Price_CPI_Less_Food_Energy_Shelter'][x],
            'trim_mean_pce': risk_data['Trimmed_Mean_PCE_Inflation_Rate'][x],
            'sixteenP_trim_mean_cpi': risk_data['16_Percent_Trimmed_Mean_CPI'][x],
            'median_cpi': risk_data['Median_CPI'][x],
            'fpcpi': risk_data['Flexible_Price_CPI'][x],
            'fpcpi_m_fe': risk_data['Flexible_Price_CPI_Less_Food_Energy'][x],
            'mean': risk_data['mean'][x],
            'user_id': 1,
            'user_data_id': 1,
        }
        # print(data)
        
        query = """INSERT INTO results (date, spcpi , spcpi_m_s , spcpi_m_fe, spcpi_m_fes , trim_mean_pce, sixteenP_trim_mean_cpi, median_cpi, fpcpi, fpcpi_m_fe, mean, created_at, updated_at, user_id, user_data_id)
                VALUES (%(date)s, %(spcpi)s , %(spcpi_m_s)s, %(spcpi_m_fe)s , %(spcpi_m_fes)s , %(trim_mean_pce)s , %(sixteenP_trim_mean_cpi)s, %(median_cpi)s , %(fpcpi)s , %(fpcpi_m_fe)s , %(mean)s , NOW(), NOW(), %(user_id)s , %(user_data_id)s);"""
        # print(query)
                
        # id = return connectToMySQL('risk_analysis').query_db(query, data)
        connectToMySQL('python_final_project').query_db(query, data)
# save_risk_data()

# NOT USED
# Create insert to OpenAi CLI data preparation tool **Notice that the -> marker was used at the end of the prompts and must be used when calling chatgpt3**
def pre_data_for_openai():
    
    # Import CSV from file folder.
    data = pd.read_csv('Final_Project/flask_app/models/recent_results.csv')
    data.drop(columns=['Unnamed: 0'], inplace=True)
    print(data)

    # Create empty dataframe
    training_data_df = pd.DataFrame('', index=range(len(data)), columns=['prompt','completion'])
    print(training_data_df)

    # Get first row of data
    # print(data.iloc[0])

    # Get first value of first row of data
    # print(data.iloc[0][0])

    # Get first column name of fist row of data
    # print(data.columns[0])

    # prompt = {'prompt': f'{data.columns[0]}={data.iloc[0][0]}, {data.columns[1]}={data.iloc[0][1]}, {data.columns[2]}={data.iloc[0][2]}, {data.columns[3]}={data.iloc[0][3]}, {data.columns[4]}={data.iloc[0][4]}, {data.columns[5]}={data.iloc[0][5]}, {data.columns[6]}={data.iloc[0][6]}, {data.columns[7]}={data.iloc[0][7]}, {data.columns[8]}={data.iloc[0][8]}',
    #             'completion': f'{data.iloc[0][9]}'}

    for row in range(len(data)):
        
        prompt = f'{data.columns[0]}={data.iloc[row][0]}, {data.columns[1]}={data.iloc[row][1]}, {data.columns[2]}={data.iloc[row][2]}, {data.columns[3]}={data.iloc[row][3]}, {data.columns[4]}={data.iloc[row][4]}, {data.columns[5]}={data.iloc[row][5]}, {data.columns[6]}={data.iloc[row][6]}, {data.columns[7]}={data.iloc[row][7]}, {data.columns[8]}={data.iloc[row][8]}, {data.columns[9]}={data.iloc[row][9]}->'
        # print(prompt)
        completion = f'{data.iloc[row][9]}'
        # print(completion)
        training_data_df.iloc[row,0] = prompt
        training_data_df.iloc[row,1] = completion

    # print(training_data_df)
    training_data_df.to_csv('Final_Project/flask_app/models/cli_data_insert.csv')
    
    # Take this file and then insert into the below website to make jsonl file and then substitute """ for "
    # https://tableconvert.com/csv-to-jsonlines
    
    # 
    csv_file = 'Final_Project/flask_app/models/cli_data_insert.csv'
    jsonl_file = 'Final_Project/flask_app/models/training_data.jsonl'
    
    # Open the CSV file and create a DictReader
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        # Open the JSONL file for writing
        with open(jsonl_file, 'w') as out_file:
            # Loop over each row in the CSV file
            for row in reader:
                # Write the dictionary as a JSON line to the file
                out_file.write(json.dumps(row) + '\n')
# pre_data_for_openai()

# NOT USED
# Insert prepared data into OpenAi CLI data preparation tool
def create_training_data():

    # Import CSV from file folder.
    data = pd.read_csv('cli_data_insert.csv')
    data.drop(columns=['Unnamed: 0'], inplace=True)
    # print(data)
    
    # Initialize an empty list to store your training data
    training_data = []

    # Iterate over each row in your DataFrame
    for index, row in data.iterrows():
        # Get the prompt text and ideal generated text from the current row
        prompt = row['prompt']
        completion = row['completion']
        
        # Create a dictionary with the prompt and completion
        data_point = {"prompt": prompt, "completion": completion}
        
        # Add the dictionary to your training data list
        training_data.append(data_point)
    
    with open("training_data.json", 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    with open("training_data.json") as f:
        for line in f:
            print(line)
# create_training_data()

# FINAL VERSION
def create_training_jsonl():

    # Import CSV from file folder.
    data = pd.read_csv('Final_Project/flask_app/models/data.csv')
    # data.drop(columns=['Unnamed: 0'], inplace=True)
    # print(data)
    # print(data.columns)
    # print(len(data.columns))

    def create_jsonl(df, filename):
        
        with jsonlines.open(filename, mode='w') as writer:
            for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t in zip(data['date'], data['Sticky_Price_CPI'], data['Sticky_Price_CPI_Risk_Score'], data['Sticky_Price_CPI_Less_Shelter'], data['Sticky_Price_CPI_Less_Shelter_Risk_Score'], data['Sticky_Price_CPI_Less_Food_Energy'], data['Sticky_Price_CPI_Less_Food_Energy_Risk_Score'], data['Sticky_Price_CPI_Less_Food_Energy_Shelter'], data['Sticky_Price_CPI_Less_Food_Energy_Shelter_Risk_Score'], data['Trimmed_Mean_PCE_Inflation_Rate'], data['Trimmed_Mean_PCE_Inflation_Rate_Risk_Score'], data['16_Percent_Trimmed_Mean_CPI'], data['16_Percent_Trimmed_Mean_CPI_Risk_Score'], data['Median_CPI'], data['Median_CPI_Risk_Score'], data['Flexible_Price_CPI'], data['Flexible_Price_CPI_Risk_Score'], data['Flexible_Price_CPI_Less_Food_Energy'], data['Flexible_Price_CPI_Less_Food_Energy_Risk_Score'], data['Monthly_Risk_Score']):
                prompt = f"On {a}, the Sticky Price CPI was {b}, Sticky Price CPI Risk Score was {c}, Sticky Price CPI Less Shelter was {d}, Sticky Price CPI Less Shelter Risk Score was {e}, Sticky Price CPI Less Food Energy was {f}, Sticky Price CPI Less Food Energy Risk Score was {g}, Sticky Price CPI Less Food Energy Shelter was {h}, Sticky Price CPI Less Food Energy Shelter Risk Score was {i}, Trimmed Mean PCE Inflation Rate was {j}, Trimmed Mean PCE Inflation Rate Risk Score was {k}, 16 Percent Trimmed Mean CPI was {l}, 16 Percent Trimmed Mean CPI Risk Score was {m}, Median CPI was {n}, Median CPI Risk Score was {o}, Flexible Price CPI was {p}, Flexible Price CPI Risk Score was {q}, Flexible Price CPI Less Food Energy was {r}, Flexible Price CPI Less Food Energy Risk Score was {s} /n/n***###***/n/n"
                completion = f" {t}"
                writer.write({'prompt': prompt, 'completion': completion})
                
    create_jsonl(data, 'Final_Project/flask_app/models/training_data_final.jsonl')
# create_training_jsonl()

# Interact with ChatGPT3
def chatgpt_interaction():
    
    # Create completion
    # https://platform.openai.com/docs/api-reference/completions?lang=python
    response = openai.Completion.create(
        model="text-curie-001",
        prompt="What is the distance to earth from sun",
        temperature=1.0,
        max_tokens=50,
        )
    # print(response)
    
    # Create Edit
    # https://platform.openai.com/docs/api-reference/completions?lang=python
    response = openai.Edit.create(
        model="text-davinci-edit-001",
        input="What day of the wek is it?",
        instruction="Fix the spelling mistakes"
    )
    # print(response)
    
    # Create an image
    # https://platform.openai.com/docs/api-reference/completions?lang=python
    response = openai.Image.create(
        prompt="A cute baby sea otter",
        n=2,
        size="1024x1024"
    )
    # print(response)
    
    # Create image edit - DID NOT WORK - Error below
    # openai.error.InvalidRequestError: Uploaded image must be a PNG and less than 4 MB.
    # https://platform.openai.com/docs/api-reference/completions?lang=python
    # response = openai.Image.create_edit(
    #     image=open("Final_Project/flask_app/models/otter.png", "rb"),
    #     mask=open("Final_Project/flask_app/models/mask.png", "rb"),
    #     prompt="A cute baby sea otter wearing a beret",
    #     n=2,
    #     size="1024x1024"
    # )
    # print(response)
    
    # Create image variation - DID NOT WORK - Error below
    # openai.error.InvalidRequestError: Uploaded image must be a PNG and less than 4 MB.
    # https://platform.openai.com/docs/api-reference/completions?lang=python
    # response = openai.Image.create_variation(
    #     image=open("Final_Project/flask_app/models/mask.png", "rb"),
    #     n=2,
    #     size="1024x1024"
    # )
    # print(response)
    
    # Create Embedding - Creates an embedding vector representing the input text.
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input="The food was delicious and the waiter..."
    )
    # print(response)
    
    # Create Moderation - Checks to see if the moderation policy for Chatgpt is being met.
    openai.Moderation.create(
        input="I want to kill them.",
        model='text-moderation-ada-001'
    )
    # print(response)
# chatgpt_interaction()

# FINAL VERSION
def train_chatgpt():
    
    # get model list
    # print(openai.Model.list()['data'][0])
    
    # print out list of models and owners
    # for item in range(len(openai.Model.list()['data'])):
    #     print(f"This is the Model: {openai.Model.list()['data'][item]['id']}")
    #     print(f"This is the Owner: {openai.Model.list()['data'][item]['owned_by']}\n")
        # print(openai.Model.list()['data'][item]['permission'])
        
    # print out one model by name
    # print(openai.Model.retrieve("curie"))
    
    # delete a model
    # print(openai.Model.delete("model_id"))
    
    # List files
    # print(openai.File.list())
    
    # # # Upload file
    # openai.File.create(
    #     file=open("Final_Project/flask_app/models/training_data_copy.jsonl", "rb"),
    #     purpose='fine-tune'
    # )   
    
    # # # # Training File
    # training_file = "file-uswEhysPPth2Pk8eBstcgjef"
    
    # openai.File.create(
    #     file=open("Final_Project/flask_app/models/validation_data.jsonl", "rb"),
    #     purpose='fine-tune'
    # )   
    
    # # Validation File
    # validation_file = "file-rgwyVrdWPKveqWj51ONW0IAj"
    
    # # List files
    # print(openai.File.list())
    
    # Delete file
    # print(openai.File.delete("file_id"))
    # print(openai.File.delete(training_file))
    # print(openai.File.delete('file-xk51zQ22CyUS4ZvaqNtWTRm4'))
    # print(openai.File.list())
    
    # Retrieve file
    # print(openai.File.retrieve("file_id"))
    
    # Retrieve file content
    # content = openai.File.download("file_id")
    # print(content)
    
    # Create fine-tuned model
    # print(openai.FineTune.create(training_file=training_file, model="curie", n_epochs=16, suffix="market-risk-analyzer"))  # validation_file=validation_file,  n_epochs=, batch_size=, learning_rate_multiplier=, prompt_loss_weight=, classification_n_classes=10, suffix="market-risk-analyzer", classification_n_classes=4, compute_classification_metrics=True, n_epochs=16
    
    # List fine tuned models
    print(openai.FineTune.list())
    
    # fine_tune_id = "ft-bx7BVM2fNSBfMu7PXYDRSarN"
    
    # Retrieve a fine tuned model
    # openai.FineTune.retrieve(id="fine_tune_id")
    # print(openai.FineTune.retrieve(id=fine_tune_id))
    
    # Immediately cancel a fine tuned job
    # print(openai.FineTune.cancel(id=fine_tune_id))
    
    # list fine tune events
    # print(openai.FineTune.list_events(id=fine_tune_id))
    
    # You only get this when the model is created successfully.  Check the fine tune events to check if it was successful.
    # fine_tuned_model = 'curie:ft-personal-2023-02-28-04-36-14'
    
    # Delete fine tuned model
    # print(openai.Model.delete(sid="curie:ft-6wUWHiD63i8p6vQ1mGETewJk"))
    # print(openai.Model.delete(sid=fine_tuned_model))
    
    # Get fine-tuned results - Needs fixed?
    # openai.FineTune.results("Finetune_job_id")
    
    #The following metrics will be displayed in your results file if you set --compute_classification_metrics:
    # For multiclass classification
    # classification/accuracy: accuracy
    # classification/weighted_f1_score: weighted F-1 score    
# train_chatgpt()

# FINAL VERSION
def using_chatgpt_model():
    
    # When using Model - To get class log probabilities you can specify logprobs=5 (for 5 classes) when using your model
    
    # print('Test')
    # print(openai.FineTune.list())
    
    fine_tune_id = "curie:ft-personal:market-risk-analyzer-2023-02-28-07-44-37"
    
    result = openai.Completion.create(
        model=fine_tune_id,
        
        # First Response
        # max_tokens=3,
        # temperature=0.0,
        # Response = " 6.333"
        # prompt="On 2/1/1984, the Sticky Price CPI was 4.471, Sticky Price CPI Risk Score was 6, Sticky Price CPI Less Shelter was 4.494, Sticky Price CPI Less Shelter Risk Score was 6, Sticky Price CPI Less Food Energy was 4.525, Sticky Price CPI Less Food Energy Risk Score was 6, Sticky Price CPI Less Food Energy Shelter was 4.586, Sticky Price CPI Less Food Energy Shelter Risk Score was 6, Trimmed Mean PCE Inflation Rate was 3.64, Trimmed Mean PCE Inflation Rate Risk Score was 6, 16 Percent Trimmed Mean CPI was 4.074, 16 Percent Trimmed Mean CPI Risk Score was 7, Median CPI was 4.229, Median CPI Risk Score was 7, Flexible Price CPI was 4.925, Flexible Price CPI Risk Score was 6, Flexible Price CPI Less Food Energy was 5.801, Flexible Price CPI Less Food Energy Risk Score was 7/n/n***###***/n/n",
        
        # Second Response
        # max_tokens=1,
        # temperature=0.0,
        # prompt="On 2/1/2023, the Sticky Price CPI was 6.333, Sticky Price CPI Risk Score was 6, Sticky Price CPI Less Shelter was 5.557, Sticky Price CPI Less Shelter Risk Score was 7, Sticky Price CPI Less Food Energy was 6.334, Sticky Price CPI Less Food Energy Risk Score was 6, Sticky Price CPI Less Food Energy Shelter was 5.000, Sticky Price CPI Less Food Energy Shelter Risk Score was 6, Trimmed Mean PCE Inflation Rate was 4.33, Trimmed Mean PCE Inflation Rate Risk Score was 6, 16 Percent Trimmed Mean CPI was 6.222, 16 Percent Trimmed Mean CPI Risk Score was 6, Median CPI was 6.5, Median CPI Risk Score was 7, Flexible Price CPI was 6.7, Flexible Price CPI Risk Score was 7, Flexible Price CPI Less Food Energy was 2.099, Flexible Price CPI Less Food Energy Risk Score was 4/n/n***###***/n/n",
        
        # Third Response
        # max_tokens=3,
        # temperature=0.0,
        # Response = " 3.0"
        # prompt='On 5/5/3030, the Sticky Price CPT Risk score was around 3, Sticky Price CPI Less Shelter Risk Score was 2, Sticky Price CPI Less Food Energy Risk Score was 3, Sticky Price CPI Less Food Energy Shelter Risk Score was 2, Trimmed Mean PCE Inflation Rate Risk Score was 3, 16 Percent Trimmed Mean CPI Risk Score was 3, Median CPI Risk Score was 3, Flexible Price CPI Risk Score was 3, Flexible Price CPI Less Food Energy Risk Score was 3/n/n***###***/n/n',
        
        # Fourth Response
        # max_tokens=3,
        # temperature=0.0,
        # Response = " 4.65" - Mean would have been 4.944
        # prompt="On 2/1/2023, the Sticky Price CPI was 3.456, Sticky Price CPI Less Shelter was 3.585, Sticky Price CPI Less Food Energy was 5.443, Sticky Price CPI Less Food Energy Shelter was 3.356, Trimmed Mean PCE Inflation Rate was 6.65, 16 Percent Trimmed Mean CPI was 5.5, Median CPI was 6.5, Flexible Price CPI was 4.32, Flexible Price CPI Less Food Energy was 5.69/n/n***###***/n/n",
        
        # Breaks if you repeat the question to many times?
        # Broke = openai.error.RateLimitError: That model is still being loaded. Please try again shortly.
        # Response = " 4.778" - Mean would have been 4.944
        # temperature=0.0,
        # max_tokens=3,
        # prompt="On 6/30/8023, the Sticky Price CPI was 3.456, Sticky Price CPI Less Shelter was 3.585, Sticky Price CPI Less Food Energy was 5.443, Sticky Price CPI Less Food Energy Shelter was 3.356, Trimmed Mean PCE Inflation Rate was 6.65, 16 Percent Trimmed Mean CPI was 5.5, Median CPI was 6.5, Flexible Price CPI was 4.32, Flexible Price CPI Less Food Energy was 5.69/n/n***###***/n/n",

        # Fifth Response
        # max_tokens=3,
        # temperature=0.0,
        # Response - " 5.778". Mean would have been 6.577
        # prompt='What will the Monthly Risk Score be on 2/2/2024 if Sticky Prices CPI is 6.543, Sticky Price CPI Less Shelter is 5.334, Sticky Price CPI Less Food Energy is 5.245, Sticky Price CPI Less Food Energy Shelter is 3.543, Trimmed Mean PCE Inflation Rate is 6.232, 16 Percent Trimmed Mean CPI is 6.5, Median CPI is 7.9, Flexible Price CPI is 10.6, Flexible Price CPI Less Food Energy is 7.3?/n/n***###***/n/n',
        
        # Follow up question
        # max_tokens=3,
        # temperature=0.0,
        # Response - "/n/n 5.778, Flexible Price CPI Less Food Energy Shelter"
        # prompt='what is the rational for response " 5.778" with prompt"What will the Monthly Risk Score be on 2/2/2024 if Sticky Prices CPI is 6.543, Sticky Price CPI Less Shelter is 5.334, Sticky Price CPI Less Food Energy is 5.245, Sticky Price CPI Less Food Energy Shelter is 3.543, Trimmed Mean PCE Inflation Rate is 6.232, 16 Percent Trimmed Mean CPI is 6.5, Median CPI is 7.9, Flexible Price CPI is 10.6, Flexible Price CPI Less Food Energy is 7.3?/n/n***###***/n/n"?',
        
        # temperature=0.0,
        # max_tokens=1000,
        # Response - "\n\nThe Monthly Risk Score is based on 12 consecutive months of data, so it is a good predictor of future inflation. The 16 Percent Trimmed Mean CPI was calculated using the same 12 months of data, but with the lagging three months removed.\n\nWhy is the 16 Percent Trimmed Mean CPI less volatile than the Monthly Risk Score?\n\nThe 16 Percent Trimmed Mean CPI is less volatile than the Monthly Risk Score because it is based on a smaller sample size.\n\nWhy is the 16 Percent Trimmed Mean CPI less volatile than the Median CPI?\n\nThe 16 Percent Trimmed Mean CPI is less volatile than the Median CPI because it is based on a smaller sample size.\n\nWhy is the 16 Percent Trimmed Mean CPI less volatile than the Flexible Price CPI?\n\nThe 16 Percent Trimmed Mean CPI is less volatile than the Flexible Price CPI because it is based on a smaller sample size.\n\nWhy is the Flexible Price CPI less volatile than the Flexible Price CPI Less Trimmed Mean?\n\nThe Flexible Price CPI Less Trimmed Mean is based on a smaller sample size.\n\nWhy is the Flexible Price CPI Less Trimmed Mean less volatile than the Flexible Price CPI Less Trimmed Mean?\n\nThe Flexible Price CPI Less Trimmed Mean is based on a smaller sample size.\n\nWhy is the Flexible Price CPI Less Trimmed Mean less volatile than the Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI?\n\nThe Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI is based on a smaller sample size.\n\nWhy is the Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less Flexible Price CPI Less Trimmed Mean Less"
        # Response - "\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 2.976, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 2.976, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 2.976, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 2.976, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 2.976, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 2.976, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 2.976, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 2.976, so, as you can see, the 16 Percent Trimmed Mean CPI is 0.166 less than the Monthly Risk Score.\n\nThe 16 Percent Trimmed Mean CPI was 3.049, so, as you can see, the 16 Percent Trimmed Mean CPI is 0."
        # Response - "\n\nThe Monthly Risk Score and Trimmed Mean PCE Inflation Rate are related, but not directly. A value of 1.0 would indicate an exact match, whereas a value of 0.0 would indicate no match. The relationship can be visualized best with a scatter plot, as shown in the following figure.\n\nWhat is the difference between Trimmed Mean PCE Inflation Rate and Median CPI?\n\nThe Trimmed Mean PCE Inflation Rate is similar to the Median CPI, but uses a different data source and performs a different calculation. The Trimmed Mean PCE Inflation Rate uses the same data as the Median CPI, but excludes data for any month in which the inflation rate was less than the rate of inflation for the previous year. As a result, the Trimmed Mean PCE Inflation Rate is similar to, but not the same as, the Median CPI. The Trimmed Mean PCE Inflation Rate was developed to provide a more meaningful comparison of inflation between different periods of time.\n\nHow do I read a value in the Trimmed Mean PCE Inflation Rate column?\n\nThe value in the Trimmed Mean PCE Inflation Rate column is the Mean of the Actual Flexible Price CPI, Trimmed Mean CPI, Flexible Price CPI Less Trimmed Mean CPI, Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI, Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trimmed Mean CPI Less Flexible Price CPI Less Trim"
        
        # temperature=0.6,
        # max_tokens=200,
        # Response = "\n\nThere is a very close correlation between the two measures. The Trimmed Mean PCE Inflation Rate is published four times a year and, in the 12 months after each release, the values for the previous four releases are averaged to form the Trimmed Mean PCE Inflation Rate. The values for the previous four releases are then averaged to form the Monthly Risk Score. Consequently, the values for the previous four releases are derived from the same source data and, as such, the values for the previous four releases are very closely correlated.\n\nWhy is there a difference between the two measures?\n\nAlthough the Monthly Risk Score is derived from the same source data as the Trimmed Mean PCE Inflation Rate, there is some additional calculation involved in producing the Monthly Risk Score. In particular, the values for the previous four releases are averaged to form the Trimmed Mean PCE Inflation Rate. The values for the previous four releases are then averaged to form the"
        # Response = "\n\nThis is a measure of the overall health of the economy, so it's not surprising that it has a close relationship with inflation. But the relationship is not linear.\n\nBy looking at the relationship between the two variables, we can see that the relationship is not perfect, and that there is a significant amount of variability around the two variables.\n\nHere is a scatter plot of the two variables:\n\nThis shows that the relationship between the two variables is not perfect. In fact, the relationship is non-linear.\n\nHere is a scatter plot of Trimmed Mean PCE Inflation Rate and Monthly Risk Score:\n\nThis shows that the relationship is not perfect. But what is interesting, is that the relationship is not linear.\n\nHere is a scatter plot of Trimmed Mean PCE Inflation Rate and Trimmed Mean PCE Inflation Rate:\n\nThis shows that the relationship is not perfect. But what is interesting, is that"
        # Response = "\n\nAlthough the Monthly Risk Score and Trimmed Mean PCE Inflation Rate are not directly comparable, they are closely related. The Monthly Risk Score is a more accurate representation of inflation risk than the Trimmed Mean PCE Inflation Rate because it is based on a larger sample of inflation rates. The Trimmed Mean PCE Inflation Rate is a more accurate representation of inflation risk than the Monthly Risk Score because it is based on a smaller sample of inflation rates.\n\nHow do Trimmed Mean PCE Inflation Rate and Median CPI compare to each other?\n\nAlthough the Trimmed Mean PCE Inflation Rate and Median CPI are not directly comparable, they are closely related. The Trimmed Mean PCE Inflation Rate is a more accurate representation of inflation risk than the Median CPI because it is based on a larger sample of inflation rates. The Median CPI is a more accurate representation of inflation risk than the Trimmed Mean PCE Inflation Rate"
        # Response = "\n\nThe two measures of inflation have a close correlation. Trimmed Mean PCE Inflation Rate is 0.944, while Monthly Risk Score is 0.879, so the two are correlated at 0.879, with a correlation coefficient of 0.929, which indicates a strong correlation.\n\nThe two measures of inflation have a close correlation. Trimmed Mean PCE Inflation Rate is 0.944, while Monthly Risk Score is 0.879, so the two are correlated at 0.879, with a correlation coefficient of 0.929, which indicates a strong correlation. How do Median CPI and Trimmed Mean PCE Inflation Rate relate to each other?\n\nMedian CPI is 0.907, while Trimmed Mean PCE Inflation Rate is 0.884, so the two are correlated at 0.884, with a correlation coefficient of 0.917, which indicates a strong correlation.\n"
        # prompt='How do Monthly Risk Score and Trimmed Mean PCE Inflation Rate relate to each other?',
        
        # temperature=0.6,
        #max_tokens=100,
        # Response = "\n\nThe correlation between Monthly Risk Score and Federal Reserve Measures is 0.77, which is significant at the 0.1 level, meaning that 77 percent of the variation in Monthly Risk Score is explained by the variation in Federal Reserve Measures.\n\nWhat is the correlation between Monthly Risk Score and 12 month Percent Change in Price?\n\nThe correlation between Monthly Risk Score and 12 month Percent Change in Price is 0.86, which is significant at the 0.1 level, meaning that 86 percent"
        # prompt='What is the correlation between Monthly Risk Score and Federal Reserver Measures?',
        
        # temperature=0.2,
        # max_tokens=100,
        # Response = "\n\nThe correlation between Monthly Risk Score and Federal Reserve Measures is 0.945, which is very strong.\n\nWhat is the correlation between Monthly Risk Score and 12 Month Rate of Change?\n\nThe correlation between Monthly Risk Score and 12 Month Rate of Change is 0.945, which is very strong.\n\nWhat is the correlation between Monthly Risk Score and Rate of Change?\n\nThe correlation between Monthly Risk Score and Rate of Change is 0.945, which is"
        # prompt='What is the correlation between Monthly Risk Score and Federal Reserver Measures?',
        
        # temperature=0.5,
        # max_tokens=100,
        # Response = " If not, why not?\n\nThe Monthly Risk Score is a good measure of Market Risk. It is a good measure of Market Risk because it provides a good indication of how volatile the market is and how much of a move we can expect from one month to the next.\n\nWhat is the relationship between the Monthly Risk Score and the Annualized Return?\n\nThe Monthly Risk Score is a good predictor of the Annualized Return. The Monthly Risk Score is a good measure of Market Risk"
        # prompt='Do you find the Monthly Risk Score to be a good measure of Market Risk and why?',
        
        # temperature=0.0,
        # max_tokens=100,
        # Response = "\n\nThe Monthly Risk Score is a good measure of Market Risk because it is a simple calculation that takes into account the current value of the Fund, the value of the Fund\u2019s holdings, and the value of the Fund\u2019s liabilities.\n\nThe Fund\u2019s value is determined by the value of its holdings and the value of its liabilities. The value of the Fund\u2019s holdings is determined by the value of the Fund\u2019s investments and the value of the"
        
        # temperature=0.0,
        # max_tokens=3,
        # Reseponse = " 4.667"
        # prompt='What is the Monthly Risk Score if Trimmed_Mean_PCE_Inflation_Rate_Risk_Score is 5.534?/n/n***###***/n/n',
        
        # temperature=0.0,
        # max_tokens=100,
        # Response = "\n\nThe Trimmed Mean PCE Inflation Rate Risk Score was 5.534, so the Monthly Trimmed Mean PCE Inflation Rate Risk Score was 6.111, so the 2nd Trimmed Mean PCE Inflation Rate Risk Score was 6.111, so the 3rd Trimmed Mean PCE Inflation Rate Risk Score was 6.111, so the 4th Trimmed Mean PCE Inflation Rate Risk Score was 6.111, so"
        # prompt='Why is the Monthly Risk Score 4.667 if Trimmed_Mean_PCE_Inflation_Rate_Risk_Score is 5.534?',
        
        # temperature=0.0,
        # max_tokens=3,
        # Response = " 5.778"
        # prompt="On 6/01/2023, the Sticky Price CPI was 6.332, Sticky Price CPI Less Shelter was 5.232, Sticky Price CPI Less Food Energy was 6.25, Sticky Price CPI Less Food Energy Shelter was 4.553, Trimmed Mean PCE Inflation Rate was 6.754, 16 Percent Trimmed Mean CPI was 7.253, Median CPI was 6.653, Flexible Price CPI was 3.563, Flexible Price CPI Less Food Energy was 7.25/n/n***###***/n/n",
        # Response = " 5.8"
        # prompt="On 6/01/2023, the Sticky Price CPI was 6.3, Sticky Price CPI Less Shelter was 5.2, Sticky Price CPI Less Food Energy was 6.2, Sticky Price CPI Less Food Energy Shelter was 4.5, Trimmed Mean PCE Inflation Rate was 6.7, 16 Percent Trimmed Mean CPI was 7.2, Median CPI was 6.6, Flexible Price CPI was 3.5, Flexible Price CPI Less Food Energy was 7.2/n/n***###***/n/n",

        # temperature=0.0,
        # max_tokens=3,
        # Response = " 2966767" - Mean = 7340049 difference of 4373282
        # prompt="On 6/01/2023, the Sticky Price CPI was 2983248, Sticky Price CPI Less Shelter was 29852025, Sticky Price CPI Less Food Energy was 29032459, Sticky Price CPI Less Food Energy Shelter was 304395, Trimmed Mean PCE Inflation Rate was 398450, 16 Percent Trimmed Mean CPI was 592052, Median CPI was 593053, Flexible Price CPI was 205925, Flexible Price CPI Less Food Energy was 2098832/n/n***###***/n/n",

        # temperature=0.0,
        # max_tokens=3,
        # Response = " 2.0" - Mean = 2.155
        # prompt="On 6/01/2023, the Sticky Price CPI was 1.3, Sticky Price CPI Less Shelter was 2.2, Sticky Price CPI Less Food Energy was 3.2, Sticky Price CPI Less Food Energy Shelter was 1.5, Trimmed Mean PCE Inflation Rate was 1.7, 16 Percent Trimmed Mean CPI was 2.2, Median CPI was 3.6, Flexible Price CPI was 2.5, Flexible Price CPI Less Food Energy was 1.2/n/n***###***/n/n",
        
        # temperature=0.0,
        # max_tokens=3,
        # Response =  " asdf***"  it took the average of the first 4 characters of each word and then **?
        # prompt="On 6/01/2023, the Sticky Price CPI was aassdd, Sticky Price CPI Less Shelter was sfdsf, Sticky Price CPI Less Food Energy was asdfads, Sticky Price CPI Less Food Energy Shelter was asdfdas, Trimmed Mean PCE Inflation Rate was sdfasf, 16 Percent Trimmed Mean CPI was asdfadsf, Median CPI was sdafds, Flexible Price CPI was asedfds, Flexible Price CPI Less Food Energy was asdfds/n/n***###***/n/n",
        
        # Response = " bscf" - Completely random?
        temperature=0.0,
        max_tokens=3,
        prompt="On 6/01/2023, the Sticky Price CPI was wejerv, Sticky Price CPI Less Shelter was ipoyumg, Sticky Price CPI Less Food Energy was zxcvcn, Sticky Price CPI Less Food Energy Shelter was erttrb, Trimmed Mean PCE Inflation Rate was zsdfdx, 16 Percent Trimmed Mean CPI was bvcbnmmg, Median CPI was ghjrfes, Flexible Price CPI was sddfred, Flexible Price CPI Less Food Energy was kjhyuh/n/n***###***/n/n",

        )
    
    print(result)
# using_chatgpt_model()