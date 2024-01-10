import pandas as pd
import csv
import json
import os
import time
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

############################################################################
###################### FILL IN THE RIGHT VALUES HERE  ######################

# input file - SPECIFY
prediction_mode_string = "_one_shot_"
data_mode = "growth" #"growth" #"value"

# system message to "prepare" the llm
val_prep_msg = "You are a prediction agent predicting the average land surface temperature for a country in a given year when provided an input detailing the country's average land surface temperature for the last 10 years. Be sure to reply with just a number, not an explanation."
growth_prep_msg = "You are a prediction agent predicting the average land surface temperature growth rate for a country in a given year when provided an input detailing the country's average land surface temperature growth rate for the last 10 years. Be sure to reply with just a number, not an explanation."

# limit how many predictions to make
PREDICTION_LIMIT = 250 # if you want to just test the code, set to 5
debug_mode = False

if PREDICTION_LIMIT < 50:
    debug_mode = True


# #
# # if working with nominal GDP - uncomment block below
# # ------------------------------------------------------------------------

# input_csv_filename = "../data/gdp/test_cleaned_gdp.csv" #'../data/gdp/cleaned_gdp_df_countries.csv'
# output_csv_filename = "test_cleaned_gdp" + prediction_mode_string + "predictions.csv"

# # Train / test splits
# test_windows_start = 2012
# predict_after_years = 10

# prep_msg = nom_prep_msg


#
# if working with GDP growth - uncomment block below
# ------------------------------------------------------------------------

# Train / test splits
prediction_horizon = 1
test_windows_start = 2012 #2012 #2010
predict_after_years = 10 #10 #12

if data_mode == "value":
    input_csv_filename = "../data/CRU/lst_by_country.csv" 
    output_csv_filename = "../data/CRU/test_cleaned_lst" + prediction_mode_string + "predictions.csv"
    prep_msg = val_prep_msg
else:
    input_csv_filename = "../data/CRU/lst_growth_by_country.csv" 
    output_csv_filename = "../data/CRU/test_cleaned_lst_growth" + prediction_mode_string + "predictions.csv"
    prep_msg = growth_prep_msg


############################################################################
############################################################################

# Read the CSV file 
with open(input_csv_filename, newline='') as csvfile:
    data = list(csv.DictReader(csvfile))

# Define the system message
system_message = prep_msg

# Initialize a list to store the results
finetuned_predictions = []
true_values = []
countries = []

#-----------------------------------------------------------------
#
# One shot prediction helper functions
#
#-----------------------------------------------------------------

'''
Wrapper
'''

def query_model(row, window_start, prep_msg, data_mode):
    if data_mode == "value":
        country_name, finetuned_prediction, true_value, input_message, task_message = query_model_lst(row, window_start)
    elif data_mode == "growth":
        country_name, finetuned_prediction, true_value, input_message, task_message = query_model_lst_growth(row, window_start)
    else:
        print(f"Error: Invalid prediction mode - choose value or growth")
    return country_name, finetuned_prediction, true_value, input_message, task_message

'''
For predicting nominal GDP
'''

def query_model_lst(row, window_start):
    country_name = row['Country'] # loop through each country
    
    # Create sliding window prompts
    i = window_start
    j = window_start - predict_after_years - 1
    prediction_year = i + predict_after_years

    # regular prompts
    input_message_example = f"Example: The country of interest is: {country_name} and average land surface temperature in degrees Celsius is {j}: {str(round(float(row[str(j)]), 2))}, {j+1}: {str(round(float(row[str(j+1)]), 2))}, {j+2}: {str(round(float(row[str(j+2)]), 2))}, {j+3}: {str(round(float(row[str(j+3)]), 2))}, {j+4}: {str(round(float(row[str(j+4)]), 2))}, {j+5}: {str(round(float(row[str(j+5)]), 2))}, {j+6}: {str(round(float(row[str(j+6)]), 2))}, {j+7}: {str(round(float(row[str(j+7)]), 2))}, {j+8}: {str(round(float(row[str(j+8)]), 2))}, {j+9}: {str(round(float(row[str(j+9)]), 2))}.\nPredict the average land surface temperature (in degrees Celsius) for {country_name} in {j+predict_after_years}:  {str(round(float(row[str(j+predict_after_years)]), 2))}\n\n"

    input_message_problem = f"Problem: The country of interest is: {country_name} and average land surface temperature in degrees Celsius is {i}: {str(round(float(row[str(i)]), 2))}, {i+1}: {str(round(float(row[str(i+1)]), 2))}, {i+2}: {str(round(float(row[str(i+2)]), 2))}, {i+3}: {str(round(float(row[str(i+3)]), 2))}, {i+4}: {str(round(float(row[str(i+4)]), 2))}, {i+5}: {str(round(float(row[str(i+5)]), 2))}, {i+6}: {str(round(float(row[str(i+6)]), 2))}, {i+7}: {str(round(float(row[str(i+7)]), 2))}, {i+8}: {str(round(float(row[str(i+8)]), 2))}, {i+9}: {str(round(float(row[str(i+9)]), 2))}."
    task_message = f"\nPredict the average land surface temperature (in degrees Celsius) for {country_name} in {prediction_year}: "

    input_message = input_message_example + input_message_problem

    # what the LLM should return
    true_value = round(float(row[str(i+predict_after_years)]), 2)
    model="gpt-3.5-turbo" #"gpt-3.5-turbo-1106" #"gpt-3.5-turbo"
    
    # messages to user about what exactly is happening
    if debug_mode:
        print(input_message)
        print(task_message)
        print("Model being used for predictions is ", model)
        print("Predictions are being made for year ", prediction_year)
        print("First year for which data is being prompted in is ", window_start)
        
    # query the model
    completion = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_message + task_message}
            ]
    )
    
    finetuned_prediction_dict = dict()
    finetuned_prediction_dict = dict(completion.choices[0].message)

    # convert returned message to a float with some error handling 
    try:
        float_value_out = float(finetuned_prediction_dict['content'])
    except ValueError:
        print(f"Error: Invalid value encountered. Could not convert '{finetuned_prediction_dict['content']}' to a float.")
        float_value_out = None  
    finetuned_prediction = float_value_out

    # preview results
    print(country_name, finetuned_prediction, true_value)

    return country_name, finetuned_prediction, true_value, input_message, task_message

'''
For predicting LST growth rates
'''

def query_model_lst_growth(row, window_start):
    '''
    to be used to query a model finetuned to make gdp growth 
    predictions in % terms on a 3 year horizon
    '''
    country_name = row['Country'] # loop through each country
    model="gpt-3.5-turbo" #"gpt-3.5-turbo-1106"#"gpt-3.5-turbo"

    # Create sliding window prompts
    i = window_start
    j = window_start - predict_after_years - 1
    prediction_year = i + predict_after_years

    # messages to user about what exactly is happening
    if debug_mode:
        print("Model being used for predictions is ", model)
        print("Predictions are being made for year ", prediction_year)
        print("First year for which data is being prompted in is ", window_start)

    # GDP growth prompts
    input_message_example = f"Example: The country of interest is: {country_name} and average land surface temperature growth rate in degrees Celsius is {j}: {str(round(float(row[str(j)]), 2))}, {j+1}: {str(round(float(row[str(j+1)]), 2))}, {j+2}: {str(round(float(row[str(j+2)]), 2))}, {j+3}: {str(round(float(row[str(j+3)]), 2))}, {j+4}: {str(round(float(row[str(j+4)]), 2))}, {j+5}: {str(round(float(row[str(j+5)]), 2))}, {j+6}: {str(round(float(row[str(j+6)]), 2))}, {j+7}: {str(round(float(row[str(j+7)]), 2))}, {j+8}: {str(round(float(row[str(j+8)]), 2))}, {j+9}: {str(round(float(row[str(j+9)]), 2))}.\nPredict the average land surface temperature growth rate (in degrees Celsius) for {country_name} in {j+predict_after_years}:  {str(round(float(row[str(j+predict_after_years)]), 2))}\n\n"

    input_message_problem = f"Problem: The country of interest is: {country_name} and average land surface temperature growth rate in degrees Celsius is {i}: {str(round(float(row[str(i)]), 2))}, {i+1}: {str(round(float(row[str(i+1)]), 2))}, {i+2}: {str(round(float(row[str(i+2)]), 2))}, {i+3}: {str(round(float(row[str(i+3)]), 2))}, {i+4}: {str(round(float(row[str(i+4)]), 2))}, {i+5}: {str(round(float(row[str(i+5)]), 2))}, {i+6}: {str(round(float(row[str(i+6)]), 2))}, {i+7}: {str(round(float(row[str(i+7)]), 2))}, {i+8}: {str(round(float(row[str(i+8)]), 2))}, {i+9}: {str(round(float(row[str(i+9)]), 2))}."
    task_message = f"\nPredict the average land surface temperature growth rate (in degrees Celsius) for {country_name} in {prediction_year}: "

    input_message = input_message_example + input_message_problem

    # input_message = f"The country of interest is: {country_name} and growth rates in percent of real GDP seen in previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
    # task_message = f"Predict the GDP growth rate (in percent) for {country_name} in {prediction_year}: "

    # what the LLM should return
    true_value = round(float(row[str(i+predict_after_years)]), 2)

    # query the model
    completion = openai.chat.completions.create(
      model=model,
      messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": input_message + task_message}
      ]
    )
    finetuned_prediction_dict = dict()
    finetuned_prediction_dict = dict(completion.choices[0].message)
    
    # convert returned message to a float with some error handling 
    try:
        float_value_out = float(finetuned_prediction_dict['content'])
    except ValueError:
        print(f"Error: Invalid value encountered. Could not convert '{finetuned_prediction_dict['content']}' to a float.")
        float_value_out = None  
    finetuned_prediction = float_value_out

    # preview results
    print(country_name, finetuned_prediction, true_value)

    return country_name, finetuned_prediction, true_value, input_message, task_message

# loop through all countries and return results
prediction_idx = 0
print("Making predictions on dataset ", input_csv_filename)
print("Writing results to ", output_csv_filename)
print("Model is being used as agent for: ", prep_msg)

for row in data:
    if prediction_idx < PREDICTION_LIMIT:
        country_name, y_hat, y_true, input_message, task_message = query_model(row, test_windows_start, prep_msg=prep_msg, data_mode=data_mode)
        time.sleep(1)
        if prediction_idx == 0:
           print(input_message)
           print(task_message)
           print("First year for which data is being prompted in is ", test_windows_start)
        countries.append(country_name)
        true_values.append(y_true)
        finetuned_predictions.append(y_hat)
    prediction_idx = prediction_idx + 1

# store results in a dataframe and write to disk
results_df = pd.DataFrame()
results_df['country'] = countries
if data_mode == "value":
    results_df['predicted_lst'] = finetuned_predictions
    results_df['observed_lst'] = true_values
else:
    results_df['predicted_lst_growth'] = finetuned_predictions
    results_df['observed_lst_growth'] = true_values
results_df.to_csv(output_csv_filename, index=False)