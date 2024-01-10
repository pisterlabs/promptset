'''
This code reads a time series dataset, preprocesses the data, and uses OpenAI's GPT-3.5-turbo model to predict the next value in the sequence. 
It evaluates the model's performance by comparing the predicted values to the actual values, calculates the Mean Absolute Percentage Error (MAPE) 
for both the AI model and a benchmark persistence model, and plots the results.'''

import pandas as pd
import numpy as np
import os
import time
import re
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

from array_similarity_search import similarity_search

#-------------------PREPARE DATA-------------------#

# Set the number of values in the sequence to predict the next value.
SEQUENCE_SIZE = 5

# Set the number of predictions to get. Make sure you allow for enough training data for the model. "80% train/20 test" split is recommended.
MAX_ITERATIONS = 500

# Set the number of the most similar question:answer pairs
MAX_EXAMPLES = 3

# Set the name of the column containing the data to be predicted.
data_column = 'Data'

# Set the name of the column containing the dates.
date_column = 'Date'

# Read the data from the CSV file.
data = pd.read_csv("Timeseries_DJI.csv")

# Drop all rows that have NaN values in the predicted column.
data = data.dropna(subset=[data_column])

# Keep only the date and data columns.
data = data[[date_column,data_column]]

# Convert the date column to a datetime object.
data[date_column]= pd.to_datetime(data[date_column])

# Sort the data by date and reset the index.
data = data.sort_values(date_column, ascending=True)
data = data.reset_index(drop=True)

# Plotting the corelation for various lags.
# The result determins what value should be assigned to the "lag" variable for the benchmark model. Usualy 1.
plot_acf(data[data_column], lags=28) 
plt.title('Correlation for different Lags')
plt.show()

# Convert the data column to a list.
data_list = data[data_column].tolist()

# Remove the elements from the beginning of the list to make the length a multiple of SEQUENCE_SIZE
data_list = data_list[len(data_list) % SEQUENCE_SIZE:]

# Define a function to convert the data into sequences.
def to_sequences(seq_size, obs, MAX_ITERATIONS):
    x, y = [], []
    for i in range(len(obs)-seq_size):
        # Create a window of data of length seq_size.
        window = obs[i:(i+seq_size)]
        
        # Get the value after the window.
        after_window = obs[i+seq_size]
        
        # Add the window and after_window to the x and y lists.
        x.append(window)
        y.append(after_window)

    # Get the the last MAX_ITERATIONS values from the list.
    sequences = x[-MAX_ITERATIONS:]
    predictions = y[-MAX_ITERATIONS:]
    # Get the last value from the predictions array.
    last_after_window = predictions[-1]

    # Get the values from MAX_ITERATIONS to the beginning of the list.
    train_sequences = x[:-MAX_ITERATIONS]
    train_predictions = y[:-MAX_ITERATIONS]

    # Return x and y numpy arrays and the last value from the y array.
    return np.array(sequences),np.array(predictions),int(last_after_window),np.array(train_sequences),np.array(train_predictions)

# Use the to_sequences function to create sequences and predictions.
sequences, predictions,last_prediction,train_sequences,train_predictions = to_sequences(SEQUENCE_SIZE, data_list,MAX_ITERATIONS)
# Convert to integers
sequences = sequences.astype(int)
predictions = predictions.astype(int)
train_sequences = train_sequences.astype(int)
train_predictions = train_predictions.astype(int)

# Print each row of the sequences and predictions
for i in range(len(sequences)):
    print("{} - {}".format(sequences[i],predictions[i]))

#-------------------CREATE TEMPLATE-------------------#

examples = []

# Define the template.
task = """You are an AI analyst and your task is to predict the next number in a sequence of numbers.
What is the next number following this sequence:{} ?
The response should be in the format of a single number without context.
Examples:{}
"""

#-------------------CREATE MODEL-------------------#

import openai

# Openai_API_key
api_key = os.environ.get('OPENAI_API_KEY')
# Configure OpenAI and Pinecone
openai.api_key = api_key

# Function to call the OpenAI API
def llm_call(
    messages: str,
    model: str = 'gpt-3.5-turbo',
    temperature: float = 0,
    max_tokens: int = 2,
):

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content.strip()
    tokens_used = response.usage.total_tokens

    return content, tokens_used

#-------------------RUN MODEL-------------------#

total_tokens_used = []
# Initialize the prompt with the first sequence.
prompt = task.format(sequences[0],examples)
# Initialize the messages list with the prompt.
messages = [{"role": "user", "content": prompt}]

iterations = 0

actual_list = []
predicted_list = []
correct_answers = []

# Main loop to run the model, and modify the prompt based on the model's response.
for i in range(0,MAX_ITERATIONS):
    print("Iteration:{}".format(i+1))

    # Call the OpenAI API.
    try:
        next_nr,tokens_used = llm_call(messages)
    except openai.error.RateLimitError:
        print(
            "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
        )
        time.sleep(10)  # Wait 10 seconds and try again
        next_nr,tokens_used = llm_call(messages)

    # Append tokens_used to total_tokens_used and sum the total tokens used.
    total_tokens_used.append(tokens_used)

    # Sanitize the LLM response.Strip any non 0 or 1 or -1 characters and spaces from the response
    next_nr = re.sub('[^\d\s-]', '', next_nr)  # Remove all non-numeric, non-space, and non-hyphen characters
    next_nr = re.sub('(?<=\d)-', '', next_nr)  # Remove hyphens that are not at the beginning of a number
    next_nr = next_nr.replace(' ', '')         # Remove spaces
    
    # Call the similarity search function to get the examples from the unseen portion of the dataset that are most similar to the predicted sequence.
    sorted_indices, sorted_target_arrays,sorted_target_results,sorted_errors = similarity_search(sequences[i],train_sequences,train_predictions,MAX_EXAMPLES)
    
    # Iterate through the sorted indices and replace the values in the examples list with the new values.
    examples = []
    for j in range(len(sorted_indices)):
        examples.append(str(sorted_target_arrays[j]) + ' => ' + str(sorted_target_results[j]))
    
    # reverse the examples list so the most similar example is last in a list of examples
    examples = examples[::-1]

    print("Examples:{}".format(examples))
    
    # if first iteration append responce to messages list.
    if i == 0:
        messages.append({"role": "assistant", "content": next_nr})
    else:
        # Replace the second element in the messages list with the model's response.
        messages[1] = {"role": "assistant", "content": next_nr}

    # If this is the last iteration, modify the sequence that will be used for prediction.
    # This is necessary if predicting on the whole timeseries dataset.
    if i == MAX_ITERATIONS-1:
        # Remove the first element from the sequences[i] and replace the last element with the predictions[i].
        seq4pred = np.delete(sequences[i], 0)
        seq4pred = np.append(seq4pred, last_prediction)
    else:
        seq4pred = sequences[i+1]

    # Compare the next_nr to the actual next number in the sequence. If the prediction is correct change the task template.
    if next_nr == str(predictions[i]):
        print("Actual:{}-Predicted:\033[92m{}\033[0m. Tokens_Used:{}".format(str(predictions[i]),next_nr,tokens_used))
        task = '''Correct! Congratulations! Now try with a new sequence:{}.
        The response must be a single number without any context.
        As an example, here are some similar Sequences => Responses:{}.'''
        # Append the sequence and the answer to the correct answers list.
        correct_answers.append([sequences[i],next_nr])
        prompt = task.format(seq4pred,examples)
    # If the prediction differs by less than 1.5% of the actual next number in the sequence, change the task template.
    elif abs(int(next_nr)-predictions[i])/predictions[i] < 0.0073:
        print("Actual:{}-Predicted:\033[93m{}\033[0m. Tokens_Used:{}".format(str(predictions[i]),next_nr,tokens_used))
        task = '''Close! You are getting better. The number that follows {} is {}. Now try with a new sequence:{}.
        The response must be a single number without any context.
        As an example, here are some similar Sequences => Responses:{}.'''
        prompt = task.format(sequences[i],predictions[i],seq4pred,examples)
    # If the prediction is incorrect change the task template.
    else:
        print("Actual:{}-Predicted:\033[91m{}\033[0m. Tokens_Used:{}".format(str(predictions[i]),next_nr,tokens_used))
        task = '''Incorrect. The number that follows {} is {}. Now try with a new sequence:{}.
        The response must be a single number without any context.
        As an example, here are some similar Sequences => Responses:{}.'''
        prompt = task.format(sequences[i],predictions[i],seq4pred,examples)


    # If first iteration append prompt to messages list.
    if i == 0:
        messages.append({"role": "user", "content": prompt})
        new_init_prompt = '''You are an AI analyst and your task is to predict the next number in a sequence of numbers.
        The response must be a single number without any context.
        '''
        messages[0] = {"role": "user", "content": new_init_prompt}
    else:
        # Replace the last element in the messages list with the prompt.
        messages[2] = {"role": "user", "content": prompt}

    print(messages)
    
    iterations += 1

    # Append the actual and predicted values to the lists.
    actual_list.append(predictions[i])
    predicted_list.append(int(next_nr))

    # If this is the last iteration, run the model for one more time to get the final prediction.
    if i == MAX_ITERATIONS-1:
        messages = messages
        next_nr,tokens_used = llm_call(messages)
        total_tokens_used.append(tokens_used)
        print("\033[94mFinal Prediction: {} => {}\033[0m".format(seq4pred,next_nr))


#-------------------EVALUATE LLM, AND GET REASONING BEHIND ANSWERS-------------------#

actual_array = np.array(actual_list)
predicted_array = np.array(predicted_list)

mape = np.mean(np.abs((actual_array - predicted_array) / actual_array)) * 100
mape_rounded = round(mape, 2)

print("\n\033[95mGPT-3.5-turbo Mean Absolute Percentage Error: {}%\033[0m".format(mape_rounded))

# Print the number of correct answers.
print("Number of correct answers: {}".format(len(correct_answers)))

# Print the list of correct answers
print("Correct answers:")
str_correct_answers = []
for i in range(0,len(correct_answers)):
    str_correct_answers.append("{} => {}".format(correct_answers[i][0],correct_answers[i][1]))
print((str_correct_answers))

# Prompt the LLM to explain the reasoning behind the correct answers.

task2 = '''Based on the following prompt: "{}" You accurately predicted the numbers following the sequences. Your predictions: {}.
Could you please explain how you arrived at these predictions? Keep your response as concise as possible.
'''

prompt = task2.format(task,correct_answers)
messages = [{"role": "user", "content": prompt}]
reasonning = llm_call(messages,max_tokens=1000)
print("\n\033[94mReasoning behind the prediction: {}\033[0m".format(reasonning[0]))
print("\nTokens used: {}".format(reasonning[1]))

#------------------TOTAL TOKENS USAGE------------------#

# Print the total tokens used.
total_tokens_used.append(reasonning[1])
print("\n\033[93mTotal tokens used: {}\033[0m".format(sum(total_tokens_used)))
print("\033[93mCost(USD): ${}\033[0m".format((sum(total_tokens_used)/1000)*0.002))

#-------------------BENCHMARK MODEL-------------------#

# calculate the size of the test set as 20% of the data set
days_back = int(len(data)*0.2)

# Benchmark - Persistence model (predict the next value from the previous value)
lag = 1
values = pd.DataFrame(data[data_column].values)
dataframe = pd.concat([values.shift(lag), values], axis=1)
dataframe.columns = ['t-lag', 't+lag']
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-days_back], X[len(X)-days_back:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# persistence model
def model_persistence(x):
    return x

# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
# Evaluate benchmark model
mape = np.mean(np.abs((test_y - predictions) / test_y)) * 100
mape_rounded = round(mape, 2)

print("\033[95mBenchmark Mean Absolute Percentage Error: {}%\033[0m".format(mape_rounded))

#-------------------PLOT THE RESULTS-------------------#

# Create a dataframe with the actual and predicted values.
df = pd.DataFrame({'Actual': actual_list, 'Predicted': predicted_list})

# Calculate the difference between actual and predicted values.
df['Difference'] = np.abs(df['Actual'] - df['Predicted'])

# Plot the difference using a scatter plot with a polynomial line of best fit, and the line plot of the actual and the predicted values.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.scatter(df.index, df['Difference'])
ax1.plot(np.poly1d(np.polyfit(df.index, df['Difference'], 1))(df.index),"r--")
ax1.set_title('Difference between Actual and Predicted Values')
ax1.set_xlabel('Index')
ax1.set_ylabel('Error')

ax2.plot(df.index, df['Actual'], color='navy', label='Actual')
ax2.plot(df.index, df['Predicted'], color='orange', label='Predicted')
ax2.set_title('Actual and Predicted Values')
ax2.set_xlabel('Index')
ax2.set_ylabel('Value')
ax2.legend()

plt.show()
