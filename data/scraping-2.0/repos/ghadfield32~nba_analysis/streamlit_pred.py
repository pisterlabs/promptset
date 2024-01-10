#streamlit run "c:/Users/ghadf/OneDrive/Desktop/Data Analytics/Python/ML/nba_w_l_prediction_models/nba_analysis/streamlit_pred.py"
#improvements:
# 1. add a button to update the model
# 2. rearrage the order of the upadted Results:
# 3. add dates to the scatter plot so we can see how each model does over time or make line chart
# 4. change matchup_id unique to only take data that the model is sure about so if there's duplicate 0's per matchup_id, drop it
# 5. Legend for Statistics that go into model
# 6. add the basketball chatbot to the streamlit app but make it so they have to input their own api key to use it until we can use it on cv application

#***essentials***
#add filter so predictions date_prediction_recorded can't be later than a games time, so we can't predict after the game has been played
#update correct vs incorrect predictions so it only does the prediction if prediction not null

#Model Improvements:
# add in preprocessing for non-tree classifiers X, now save it and the one_hot_encoder for non-tree's and implement in here for non-tree models
# 1's: SVM, Logistic Regression, ridge classifier, sgd classifier are all very similar in accuracy.  We should try to combine them into one model
# 0's: gaussianNB, KNN, 
# 1's/0's: RandomF, XGBoost, AdaBoost, GradientBoost, DecisionTree

# add in linear regression models for how each team will do for each statistic, maybe even change the averages to these predicted stats if they are good enough
# add in CNN for image recognition of the players
# add in LSTM for time series analysis of how each team does over time
# add in a chatbot that will give you the prediction for the game you ask it about
# add CNN to predict who will win the championship based on the players on the team


#unseen data will only pull in data until tomorrow
# 23_24_season_games_past.csv will provide the actual results for the games that have already been played

# File paths and feature names
tree_pred_path = 'data/tree_season_pred.csv'
non_tree_pred_path = 'data/non_tree_season_pred.csv'
ltsm_pred_path = 'data/ltsm_season_pred.csv'
ltsm_seq_pred_path = 'data/ltsm_seq_season_pred.csv'
past_results_path = 'data/23_24_current_season_prediction_tracker.csv'
votes_data_path = 'data/voter_pred.csv'
all_data_path = 'data/new_season_predictions/predictions_combined.csv'


import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score, mean_absolute_error
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import openai
import os

#Take out predictions with the Same value for model so we only track valuable predictions
def validate_predictions(data):
    prediction_columns = [
        'XGBoost_PREDICTION', 'Decision Tree_PREDICTION', 'Random Forest_PREDICTION', 
        'Gradient Boosting_PREDICTION', 'AdaBoost_PREDICTION', 'MLP Classifier_PREDICTION', 
        'K-Neighbors Classifier_PREDICTION', 'SVM_PREDICTION', 'SGD Classifier_PREDICTION', 
        'Ridge Classifier_PREDICTION', 'Logistic Regression_PREDICTION', 'ltsm_PREDICTION', 'ltsm_seq_PREDICTION'
    ]
    
    # Iterate over each prediction column to validate
    for col in prediction_columns:
        # Calculate the sum of predictions for each matchup within the column
        data[f'{col}_sum'] = data.groupby(['Date', 'MATCHUP_ID'])[col].transform('sum')
        
        # Identify rows where the sum of predictions is not equal to 1
        invalid_mask = data[f'{col}_sum'] != 1
        
        # Set predictions to NaN for rows where the sum is not 1
        data.loc[invalid_mask, col] = np.nan
    
    # Drop the temporary sum columns
    sum_columns = [f'{col}_sum' for col in prediction_columns]
    data = data.drop(columns=sum_columns)
    
    return data


# Data Loading Functions
def load_tree_data(path):
    return pd.read_csv(path)[['Date', 'MATCHUP_ID', 'TEAM_NAME', 'XGBoost_PREDICTION', 'Decision Tree_PREDICTION', 'Random Forest_PREDICTION', 'Gradient Boosting_PREDICTION', 'AdaBoost_PREDICTION']]

def load_non_tree_data(path):
    return pd.read_csv(path)[['Date', 'MATCHUP_ID', 'TEAM_NAME', 'MLP Classifier_PREDICTION', 'K-Neighbors Classifier_PREDICTION', 'SVM_PREDICTION', 'SGD Classifier_PREDICTION', 'Ridge Classifier_PREDICTION', 'Logistic Regression_PREDICTION']]

def load_ltsm_data(path):
    ltsm_data = pd.read_csv(path)
    ltsm_data = ltsm_data.rename(columns={'PREDICTION': 'ltsm_PREDICTION'})
    return ltsm_data[['Date', 'MATCHUP_ID', 'TEAM_NAME', 'ltsm_PREDICTION']]

def load_ltsm_seq_data(path):
    ltsm_seq_data = pd.read_csv(path)
    ltsm_seq_data = ltsm_seq_data.rename(columns={'PREDICTION': 'ltsm_seq_PREDICTION'})
    return ltsm_seq_data[['Date', 'MATCHUP_ID', 'TEAM_NAME', 'ltsm_seq_PREDICTION']]

def load_past_results(path):
    return pd.read_csv(path)

# Load voter data and remove the index column if it exists
try:
    votes_data = pd.read_csv(votes_data_path).drop(columns=['Unnamed: 0'], errors='ignore')
except FileNotFoundError:
    votes_data = pd.DataFrame(columns=['Date', 'MATCHUP_ID', 'TEAM_NAME', 'Votes'])

# Aggregate votes by Date, MATCHUP_ID, and TEAM_NAME
aggregated_votes = votes_data.groupby(['Date', 'MATCHUP_ID', 'TEAM_NAME']).sum().reset_index().rename(columns={'Votes': 'voter_predictions'})
#print(aggregated_votes.head())

# Merging function
def merge_data(tree_data, non_tree_data, ltsm_seq_data, ltsm_data):
    # Perform the merges
    tree_non_tree = pd.merge(tree_data, non_tree_data, on=['Date', 'MATCHUP_ID', 'TEAM_NAME'], how='left')
    tree_non_tree_ltsm = pd.merge(tree_non_tree, ltsm_seq_data, on=['Date', 'MATCHUP_ID', 'TEAM_NAME'], how='left')
    all_data = pd.merge(tree_non_tree_ltsm, ltsm_data, on=['Date', 'MATCHUP_ID', 'TEAM_NAME'], how='left')
    
    # Validate predictions before merging with votes
    validated_data = validate_predictions(all_data)
    
    # Now merge with votes data
    all_data_with_votes = pd.merge(validated_data, aggregated_votes, on=['Date', 'MATCHUP_ID', 'TEAM_NAME'], how='left')
    
    return all_data_with_votes



all_data = pd.read_csv(all_data_path)

# Function to calculate the daily correct percentage
def calculate_daily_correct_percentage(data, model):
    data['correct'] = data[f'{model}_PREDICTION'] == data['WL_encoded']
    daily_correct = data.groupby('Date')['correct'].mean().reset_index()
    return daily_correct.sort_values('Date')

#*******************************chatbot add on***********************************
# Initialize session state for messages at the top level to ensure it's always done.
st.session_state.setdefault("messages", [{"role": "system", "content": "Warming up on the court! 🏀 Ready to assist and share some hoops wisdom. Pass the ball, and let's get this conversation rolling!"}])

# Set the OpenAI API key.
openai.api_key = st.secrets["openai_key"]


# Function to generate initial prompt for the chatbot based on the model's predictions
def generate_initial_prompt(past_data_with_predictions):
    # You could summarize the data or just take the most recent predictions
    prompt = "Ask me anything about the NBA before 2021 or anything about LSTM Models"
    #latest_predictions = past_data_with_predictions.iloc[-1]  # Assuming the latest predictions are at the end
    #prompt = f"Today is {datetime.now().strftime('%Y-%m-%d')}. Here are the latest NBA game predictions: "
    #prompt += f"{latest_predictions['TEAM_NAME']} prediction is {latest_predictions['ltsm_seq_PREDICTION']} based on the Chronos Predictor. "
    #prompt += "What do you think about these predictions?"
    return prompt

def chatbot_sidebar(initial_prompt):
    # Initialize session state for chat messages if not already done
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [{"role": "system", "content": initial_prompt}]

    with st.sidebar:
        st.title("🏀 NBA Chatbot")
        # Display previous messages
        for msg in st.session_state["chat_messages"]:
            st.text_area(label=msg["role"], value=msg["content"], height=100, disabled=True)

        # User input for the chatbot
        user_input = st.text_input("Ask the chatbot about the NBA predictions:")

        if user_input:
            # Append user's message to the messages list
            st.session_state["chat_messages"].append({"role": "user", "content": user_input})

            # Request a completion from the model (this should be replaced with your own API call)
            # Note: You'll need to handle API calls and errors appropriately
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=st.session_state["chat_messages"]
            )
            response_text = response.choices[0].message["content"].strip()

            # Append the model's response to the messages list
            st.session_state["chat_messages"].append({"role": "assistant", "content": response_text})

            # Display the response
            st.text_area(label="Assistant", value=response_text, height=100, disabled=True)

#*******************************chatbot add on***********************************

def main():
    st.title("Man vs Machine: NBA Predictions")

    # Descriptive introduction
    st.markdown("""
        Welcome to the NBA Predictions app, where the power of human intuition meets the precision of machine learning. 
        Here, we feature two unique Long-Short-Term-Models (LSTM) that predict the outcomes of NBA games.:
        
        - **Chronos Predictor**: A LSTM model that leverages the sequence of the last 5 games to capture the momentum and dynamics of NBA teams. This model understands that the context of previous games can be vital in determining the outcome of the next game.
        
        - **Aeolus Forecaster**: A standard LSTM model that provides predictions based on current game data without the sequence memory of past games. It's named after the Greek deity of wind, symbolizing the swift and dynamic nature of its predictions.
        
        - **Regression Forecaster**: A linear regression model that predicts the outcome of NBA games based on how many points will be scored. This model is named after the type of  Recurrent Neural Network (RNN) that is uses
                
        Compare these advanced AI predictions with human votes to see if man or machine has the upper hand in predicting the outcomes of NBA games.
    """)

    
    # Load and merge data
    tree_data = load_tree_data(tree_pred_path)
    non_tree_data = load_non_tree_data(non_tree_pred_path)
    ltsm_data = load_ltsm_data(ltsm_pred_path)
    ltsm_seq_data = load_ltsm_seq_data(ltsm_seq_pred_path)
    #all_data = merge_data(tree_data, non_tree_data, ltsm_data, ltsm_seq_data)
    all_data = pd.read_csv(all_data_path)
    past_results = load_past_results(past_results_path)
    # Sidebar for navigation
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Man vs Machine", "All Predictions", "Voter Predictions"])
    
    if app_mode == "Voter Predictions":
        st.subheader('Upcoming NBA Games (LSTM Predictions)')
        
        # Load previous votes if available
        if 'votes_data' not in st.session_state:
            try:
                st.session_state['votes_data'] = pd.read_csv(votes_data_path)
            except FileNotFoundError:
                st.session_state['votes_data'] = pd.DataFrame(columns=['MATCHUP_ID', 'TEAM_NAME', 'Votes'])
        
        today_date = datetime.now().strftime('%Y-%m-%d')  # Get today's date
        selected_date = today_date  # Allow voting only for today's games
        matchups_on_selected_date = ltsm_data[ltsm_data['Date'] == selected_date]

        # Group by matchup_id to show both teams in the same section
        grouped = matchups_on_selected_date.groupby('MATCHUP_ID')

        for matchup_id, group in grouped:
            for _, matchup in group.iterrows():
                st.write(f"{matchup['TEAM_NAME']}: LSTM: {matchup['ltsm_PREDICTION']} LSTM:")

            # Voting
            selected_team = st.selectbox(f"Who will win {matchup_id}?", group['TEAM_NAME'].tolist())

            if st.button(f"Vote for {selected_team}"):
                # Iterate through the teams in the matchup
                for team_name in group['TEAM_NAME'].tolist():
                    # Record a '0' for the selected team and a '1' for the other team
                    vote_value = 0 if team_name == selected_team else 1
                    # Find the existing vote record for this matchup and team
                    existing_vote = st.session_state['votes_data'].loc[
                        (st.session_state['votes_data']['MATCHUP_ID'] == matchup_id) & 
                        (st.session_state['votes_data']['TEAM_NAME'] == team_name) & 
                        (st.session_state['votes_data']['Date'] == selected_date)
                    ]
                    
                    # Check if the team already has a vote record for this matchup on this date
                    if existing_vote.empty:
                        # If no existing vote, append the new row
                        new_row = {'Date': selected_date, 'MATCHUP_ID': matchup_id, 'TEAM_NAME': team_name, 'Votes': vote_value}
                        st.session_state['votes_data'] = pd.concat([st.session_state['votes_data'], pd.DataFrame([new_row])], ignore_index=True)
                    else:
                        # If vote exists, check if it is the same as before
                        if existing_vote['Votes'].iloc[0] != vote_value:
                            # If different, update the vote
                            st.session_state['votes_data'].loc[existing_vote.index, 'Votes'] = vote_value
                            st.success(f"Your vote for {team_name} has been updated.")
                        else:
                            # If the same, inform the user
                            st.info(f"You have already voted for {team_name} as {'winning' if vote_value == 0 else 'losing'} this matchup.")

                # Save the updated votes to CSV
                st.session_state['votes_data'].to_csv(votes_data_path, index=False)
                st.rerun()  # Rerun the app to reflect the updated votes

            st.write("---")

    
    elif app_mode == "All Predictions":
        st.subheader('All Predictions')
        #sort by date and reset index
        all_data = all_data.sort_values(by=['Date'], ascending=False)
        all_data = all_data.reset_index(drop=True)
        st.write(all_data)

    if app_mode == "Man vs Machine":
        st.subheader('Man vs Machine: Model Performance')

        models = [
            "XGBoost", "Decision Tree", "Random Forest",
            "Gradient Boosting", "AdaBoost", "MLP Classifier",
            "K-Neighbors Classifier", "SVM", "SGD Classifier",
            "Ridge Classifier", "Logistic Regression", "ltsm",
            "ltsm_seq", "linreg_team_point", "linreg_wl"
        ]
        accuracies = {}
        mae_values = {}

        for model in models:
            # Filter out NaN values for both predictions and actual results
            filtered_df = all_data.dropna(subset=['WL_encoded', f"{model}_PREDICTION"])
            # For the linear regression model, calculate MAE
            if "linreg_team_point" in model:
                mae_values[model] = mean_absolute_error(filtered_df['PTS'], filtered_df[f"{model}_PREDICTION"])
            else:
                # For classification models, calculate accuracy
                accuracies[model] = accuracy_score(filtered_df['WL_encoded'], filtered_df[f"{model}_PREDICTION"])

        # Combine the accuracy and MAE dictionaries into a dataframe
        accuracy_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy'])
        mae_df = pd.DataFrame(list(mae_values.items()), columns=['Model', 'MAE'])

        # Merge the two dataframes on the 'Model' column
        performance_df = pd.merge(accuracy_df, mae_df, on='Model', how='outer')

        # If you want to display 'linreg' as 'Linear Regression' in your table
        performance_df['Model'] = performance_df['Model'].replace({
            'linreg_team_point': 'Linear Regression (Team Points)',
            'linreg_wl': 'Linear Regression (Win/Loss)',
            'ltsm': 'Chronos Predictor',
            'ltsm_seq': 'Aeolus Forecaster'
        })
        #change Accuracy column into percentages
        performance_df['Accuracy'] = performance_df['Accuracy'].map("{:.2%}".format)

        #change MAE to 2 decimal places
        performance_df['MAE'] = performance_df['MAE'].map("{:.2f}".format)

        # reorder columns so included models come first in the table
        included_models = ['Chronos Predictor', 'Aeolus Forecaster', 'Linear Regression (Win/Loss)', 'Linear Regression (Team Points)']
        performance_df = performance_df.reindex(performance_df['Model'].isin(included_models).sort_values(ascending=False).index)

        st.write(performance_df)

        included_models = ['ltsm', 'ltsm_seq', 'linreg_wl']
        all_model_data = pd.DataFrame()

        for model in included_models:
            model_data = all_data.dropna(subset=['WL_encoded', f"{model}_PREDICTION"])
            model_data['Model'] = model  # Add a column for the model name
            daily_correct_percentage = calculate_daily_correct_percentage(model_data, model)
            daily_correct_percentage['Model'] = model  # Add a column for the model name
            all_model_data = pd.concat([all_model_data, daily_correct_percentage], ignore_index=True)

        # Using Plotly to create an interactive plot
        fig = px.scatter(all_model_data, x='Date', y='correct', color='Model',
                         title='Daily Correct Percentage of Included Models',
                         labels={'correct': 'Correct Percentage'},
                         hover_data={'Model': True})

        fig.update_traces(mode='lines+markers')
        fig.update_layout(legend_title_text='Model')
        fig.update_yaxes(tickformat=".1%")  # Format the y-axis ticks as percentages

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Generate initial prompt for the chatbot
        initial_prompt = generate_initial_prompt(performance_df)

        # Display the chatbot in the sidebar
        chatbot_sidebar(initial_prompt)
        

# Run the main function
if __name__ == "__main__":
    main()


