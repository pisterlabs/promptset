import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from openai import OpenAI
import time

# Initialize your LLM client
local_server = "http://localhost:12345/v1"
client = OpenAI(base_url=local_server, api_key="sk_1234567890")

#* Create config file in the future
BOOL_SYSTEM_MESSAGE = """You are excellent message moderator, expert in detecting fraudulent messages.

You will be given "Messages" and your job is to predict if a message is fraudulent or not.

You ONLY respond FOLLOWING this json schema:

{
    "is_fraudulent": {
        "type": "boolean",
        "description": "Whether the message is predicted to be fraudulent."
    }
}

You MUST ONLY RESPOND with a boolean value in JSON. Either true or false in JSON. NO EXPLANATIONS OR COMMENTS.

Example of valid responses:
{
    "is_fraudulent": true
}
or 
{
    "is_fraudulent": false
}
"""


# Choose which system message to use based on your requirement
system_message = BOOL_SYSTEM_MESSAGE

def predict_fraudulence_modified(sms_text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": sms_text},
            ],
            temperature=0.3,
            max_tokens=100,
        )
        prediction_content = response.choices[0].message.content
        try:
            prediction_json = json.loads(prediction_content)
            if "is_fraudulent" in prediction_json and type(prediction_json["is_fraudulent"]) == bool:
                return prediction_json['is_fraudulent'], 'valid'
            else:
                # Record the raw response if JSON is invalid but well-formed
                return prediction_content, 'invalid_json'
        except json.JSONDecodeError:
            # Record the raw response if JSON structure is invalid
            return prediction_content, 'invalid_json_structure'
    except Exception as e:
        return str(e), 'error'


# Load the dataset
try:
    df_clone = pd.read_csv('../dataset/parallel/processing/sms_predictions_3_in_progress.csv')
    print("Resuming from saved progress.")
except FileNotFoundError:
    df = pd.read_csv('../dataset/parallel/sms_3.csv')
    df_clone = df.copy()
    df_clone['Predicted'] = None
    df_clone['PredictionType'] = None
    print("Starting from the beginning.")

# Start the timer
start_time = time.time()

# Variables for tracking progress and time
total_messages = len(df_clone)
messages_processed = 0
time_per_batch = 10  # Update time tracking every 100 messages

# Iterate over each SMS message that hasn't been processed yet
for index, row in df_clone.iterrows():
    if pd.isnull(row['Predicted']):
        sms_text = row['SMS test']
        
        # Get prediction from LLM
        predicted_label, prediction_type = predict_fraudulence_modified(sms_text)

        # Record prediction and type
        df_clone.at[index, 'Predicted'] = predicted_label
        df_clone.at[index, 'PredictionType'] = prediction_type

        # Save progress after each message
        df_clone.to_csv('../dataset/parallel/processing/sms_predictions_3_in_progress.csv', index=False)
        
        messages_processed += 1
        
        # Time tracking and progress update
        if messages_processed % time_per_batch == 0 or messages_processed == total_messages:
            current_time = time.time()
            elapsed_time = current_time - start_time
            avg_time_per_message = elapsed_time / messages_processed
            estimated_total_time = avg_time_per_message * total_messages
            estimated_time_remaining = estimated_total_time - elapsed_time
            progress_percentage = (messages_processed / total_messages) * 100

            print(f"Processed {messages_processed}/{total_messages} messages ({progress_percentage:.2f}% complete).")
            print(f"Average time per message: {avg_time_per_message:.2f} seconds.")
            print(f"Estimated time remaining: {estimated_time_remaining // 60:.0f} minutes and {estimated_time_remaining % 60:.0f} seconds.")

# Final save after all messages are processed
df_clone.to_csv('../dataset/parallel/results/sms_predictions_3_results.csv', index=False)
print("All messages processed and final results saved.")

# Calculate and print metrics (only for valid predictions)
valid_predictions_df = df_clone[df_clone['PredictionType'] == 'valid']
accuracy = accuracy_score(valid_predictions_df['Fraudulent'], valid_predictions_df['Predicted'])
precision = precision_score(valid_predictions_df['Fraudulent'], valid_predictions_df['Predicted'])
recall = recall_score(valid_predictions_df['Fraudulent'], valid_predictions_df['Predicted'])
f1 = f1_score(valid_predictions_df['Fraudulent'], valid_predictions_df['Predicted'])

print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}")
