import os
from openai import OpenAI
import json
from datetime import datetime
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
import csv
import time
import pickle
import pandas as pd
from models import Customer_segment, Channel, Campaign, Target_audience

# *******************************************************************************************************************************
# Load models

# Path to the linear regression model
lr_model_path = 'prediction_models/marketing_planner_linear_regression_model.sav'
with open(lr_model_path, 'rb') as file:
    model_lr = pickle.load(file)

# Path to the ridge model
ridge_model_path = 'prediction_models/marketing_planner_ridge_model.sav'
with open(ridge_model_path, 'rb') as file:
    model_ridge = pickle.load(file)


# *******************************************************************************************************************************
# Process input data

def process_input_data(df):
    """Process input data"""

    # Parse 'Date' and decompose it into 'year', 'month', and 'day'
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['year'] = df['start_date'].dt.year
    df['month'] = df['start_date'].dt.month
    df['day'] = df['start_date'].dt.day
    df.drop(['start_date','name'], axis=1, inplace=True)

    # One-hot encode categorical variables
    categorical_columns = ['target_audience',
                        'customer_segment']
    processed_df = pd.get_dummies(df, columns=categorical_columns)


    # Define expected columns
    # Get list of customer segment names
    customer_segments = Customer_segment.query.all()
    customer_segment_names = ["customer_segment_"+segment.name for segment in customer_segments]

    # Get list of channel names
    channels = Channel.query.all()
    channel_names = [channel.name for channel in channels]

    # Get list of target audiences
    target_audiences = Target_audience.query.all()
    target_audience_names = ["target_audience_"+target_audience.name for target_audience in target_audiences]

    # Combine all expected columns

    expected_columns = ['duration',
                        'spend_email',
                        'spend_facebook',
                        'spend_google_ads',
                        'spend_instagram',
                        'spend_website',
                        'spend_youtube',
                        'year',
                        'month',
                        'day',] + customer_segment_names + target_audience_names

    # Ensure all expected columns are present, adding missing ones as needed
    for col in expected_columns:
        if col not in df.columns:
            df[col] = False 

    # Reorder columns to match expected format
    processed_df = df[expected_columns]

    print("***PROCESSED DATA:***", processed_df)

    return processed_df


# *******************************************************************************************************************************
# Make predictions

def make_predictions(data):
    """Make predictions on data"""

    # Make predictions
    prediction_lr = model_lr.predict(data).tolist()[0]
    prediction_ridge = model_ridge.predict(data).tolist()[0]


    prediction_mean = []

    for i in range(len(prediction_lr)):
        prediction_mean.append((prediction_lr[i] + prediction_ridge[i]) / 2)
    
    # turn list into a dictionary with keys as channel names

    projected_columns = ['projected_revenue_email', 
                          'projected_revenue_facebook', 
                          'projected_revenue_google_ads', 
                          'projected_revenue_instagram', 
                          'projected_revenue_website', 
                          'projected_revenue_youtube']
    
    # prediction_mean_dict = dict(zip(projected_columns, prediction_mean))
    prediction_mean_dict = dict(zip(projected_columns, prediction_ridge))
    

    print("***PREDICTIONS:***", prediction_mean_dict)
    # prediction_mean = (prediction_lr + prediction_rf) / 2

    # calculate ROI
    # investment = data['spend_total'].tolist()[0]
    # roi = (prediction_lr - investment) / investment

    # Return predictions and ROI
    return prediction_mean_dict