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
gdp_mode = "growth" #"growth" #"nominal"

# system message to "prepare" the llm
nom_prep_msg = "You are a prediction agent predicting the GDP for a country in a given year when provided an input detailing the country's GDP for the last 10 years and showing an previous pattern of how GDP for that country changed over time. Be sure to reply with just a number, not an explanation."
growth_prep_msg = "You are a prediction agent predicting the GDP growth rates for a country in a given year when provided an input detailing the country's GDP growth rates for the last 10 years and showing an previous pattern of how GDP growth rates for that country changed over time. Be sure to reply with just a number, not an explanation."

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

input_csv_filename = "../data/gdp/rounded_cleaned_gdp_growth_df.csv" 
output_csv_filename = "test_cleaned_gdp_growth" + prediction_mode_string + "predictions_" + str(prediction_horizon) + "yr.csv"

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

def query_model(row, window_start, prep_msg=growth_prep_msg, gdp_mode=gdp_mode):
    if gdp_mode == "nominal":
        country_name, finetuned_prediction, true_value, input_message, task_message = query_model_gdp(row, window_start, prep_msg=growth_prep_msg)
    elif gdp_mode == "growth":
        country_name, finetuned_prediction, true_value, input_message, task_message = query_model_gdp_growth(row, window_start, prep_msg=growth_prep_msg)
    else:
        print(f"Error: Invalid prediction mode - choose nominal or growth")
    return country_name, finetuned_prediction, true_value, input_message, task_message

'''
For predicting nominal GDP
'''

def query_model_gdp(row, window_start, prep_msg):
    country_name = row['country'] # loop through each country
    
    # Create sliding window prompts
    i = window_start
    j = window_start - predict_after_years - 1
    prediction_year = i + predict_after_years

    # regular GDP prompts
    input_message_example = f"Example: The country of interest is: {country_name} and GDP (in Billions of US dollars at present day prices) from previous years is {j}: {row[str(j)]}, {j+1}: {row[str(j+1)]}, {j+2}: {row[str(j+2)]}, {j+3}: {row[str(j+3)]}, {j+4}: {row[str(j+4)]}, {j+5}: {row[str(j+5)]}, {j+6}: {row[str(j+6)]}, {j+7}: {row[str(j+7)]}, {j+8}: {row[str(j+8)]}, {j+9}: {row[str(j+9)]}.\nPredict the GDP (in Billions of US dollars at present day prices) for {country_name} in {j+predict_after_years}:  {row[str(j+predict_after_years)]}\n"

    input_message_problem = f"Problem: The country of interest is: {country_name} and GDP (in Billions of US dollars at present day prices) from previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}."
    task_message = f"Predict the GDP (in Billions of US dollars at present day prices) for {country_name} in {prediction_year}: "

    input_message = input_message_example + input_message_problem

    # what the LLM should return
    true_value = float(row[str(i+predict_after_years)])
    model="gpt-3.5-turbo" #"gpt-3.5-turbo-1106" #"gpt-3.5-turbo"
    
    # messages to user about what exactly is happening
    if debug_mode:
        print(input_message)
        print(task_message)
        print("Model being used for predictions is ", model)
        print("Predictions are being made for year ", prediction_year)
        print("First year for which data is being prompted in is ", window_start)
        
    # query the model
    completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_message + task_message}
            ]
    )
    
    finetuned_prediction_dict = dict()
    finetuned_prediction_dict = completion.choices[0].message

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
For predicting GDP growth rates
'''

def query_model_gdp_growth(row, window_start, prep_msg=growth_prep_msg):
    '''
    to be used to query a model finetuned to make gdp growth 
    predictions in % terms on a 3 year horizon
    '''
    country_name = row['country'] # loop through each country
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
    input_message_example = f"Example: The country of interest is: {country_name} and growth rates in percent of real GDP seen in previous years is {j}: {row[str(j)]}, {j+1}: {row[str(j+1)]}, {j+2}: {row[str(j+2)]}, {j+3}: {row[str(j+3)]}, {j+4}: {row[str(j+4)]}, {j+5}: {row[str(j+5)]}, {j+6}: {row[str(j+6)]}, {j+7}: {row[str(j+7)]}, {j+8}: {row[str(j+8)]}, {j+9}: {row[str(j+9)]}.\nPredict the GDP growth rate (in percent) for {country_name} in {j+predict_after_years}:  {row[str(j+predict_after_years)]}\n"

    input_message_problem = f"Problem: The country of interest is: {country_name} and growth rates in percent of real GDP seen in previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}."
    task_message = f"Predict the GDP growth rate (in percent) for {country_name} in {prediction_year}: "

    input_message = input_message_example + input_message_problem

    # input_message = f"The country of interest is: {country_name} and growth rates in percent of real GDP seen in previous years is {i}: {row[str(i)]}, {i+1}: {row[str(i+1)]}, {i+2}: {row[str(i+2)]}, {i+3}: {row[str(i+3)]}, {i+4}: {row[str(i+4)]}, {i+5}: {row[str(i+5)]}, {i+6}: {row[str(i+6)]}, {i+7}: {row[str(i+7)]}, {i+8}: {row[str(i+8)]}, {i+9}: {row[str(i+9)]}"
    # task_message = f"Predict the GDP growth rate (in percent) for {country_name} in {prediction_year}: "

    # what the LLM should return
    true_value = float(row[str(i+predict_after_years)])

    # query the model
    completion = openai.ChatCompletion.create(
      model=model,
      messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": input_message + task_message}
      ]
    )
    finetuned_prediction_dict = dict()
    finetuned_prediction_dict = completion.choices[0].message
    
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
        country_name, y_hat, y_true, input_message, task_message = query_model(row, test_windows_start, prep_msg=prep_msg, gdp_mode=gdp_mode)
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
results_df['predicted_gdp'] = finetuned_predictions
results_df['observed_gdp'] = true_values
results_df.to_csv(output_csv_filename, index=False)