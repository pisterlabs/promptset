import os
import pandas as pd

# for google
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import gspread
from google.oauth2 import service_account

# for reg
from langchain.llms import OpenAI

# for emails
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# The 'get_tipper_picks' function calculates the odds threshold for both home and away teams and then selects the home and away teams based on their predicted results.
def get_tipper_picks(predictions, prod_run=False):
    
    # Calculate odds thresholds for home and away teams
    predictions['home_odds_thresh'] = 1 / predictions['home_team_win_prob']
    predictions['away_odds_thresh'] = 1 / predictions['home_team_lose_prob'] 
    
    # Select home teams that are predicted to win and rename the columns accordingly.
    home_picks = predictions[predictions['home_team_result'] == 'Win'][['team_home', 'team_head_to_head_odds_home', 'home_odds_thresh']].copy()
    home_picks.rename(columns={'team_home': 'team', 'team_head_to_head_odds_home': 'price', 'home_odds_thresh': 'price_min'}, inplace=True)
    
    # Select away teams that are predicted to lose and rename the columns accordingly.
    away_picks = predictions[predictions['home_team_result'] == 'Loss'][['team_away', 'team_head_to_head_odds_away', 'away_odds_thresh']].copy()
    away_picks.rename(columns={'team_away': 'team', 'team_head_to_head_odds_away': 'price', 'away_odds_thresh': 'price_min'}, inplace=True)
    
    # Concatenate the home and away picks and filter rows where 'price' is more than 15% of 'price_min'.
    tipper_picks = pd.concat([home_picks, away_picks])
    tipper_picks = tipper_picks[tipper_picks['price'] > (tipper_picks['price_min'] * 1.05)]

    return tipper_picks

# The 'upload_df_to_drive' function uploads a pandas DataFrame to Google Drive as a CSV file.
def upload_df_to_drive(df, json_path, folder_id, filename):

    # Load the credentials from the service_account.json
    creds = service_account.Credentials.from_service_account_file(json_path)

    # Build the Google Drive service
    drive_service = build('drive', 'v3', credentials=creds)

    # Save your dataframe to CSV
    df.to_csv(filename, index=False)

    # Upload the file
    file_metadata = {
        'name': f"round{df['round_id'].unique()[0]}_{df['competition_year'].unique()[0]}.csv",
        'parents': [folder_id]
    }
    media = MediaFileUpload(filename, mimetype='text/csv')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print('File ID:', file.get('id'))

    # Delete the local file after upload
    os.remove(filename)

# The 'generate_reg_regan_email' function generates an email content with the help of an AI language model (OpenAI). The email contains a synopsis of NRL games and some value tips.
def generate_reg_regan_email(predictions, tipper_picks, api_key, folder_url):

    # Set up the OpenAI model using provided API key and model parameters
    llm = OpenAI(openai_api_key=api_key,
                 model_name="gpt-3.5-turbo-16k",
                 max_tokens=15000,
                 temperature=1.1)

    # Generate input_predictions string by iterating over 'predictions' dataframe and formatting data into string
    input_predictions = ""
    for index, row in predictions.iterrows():
        input_predictions += f"""
            Round Name: {row['round_name']},
            Home Team Result: {row['home_team_result']},
            Home Team: {row['team_home']}, 
            Home Team Position: {row['position_home']},
            Home Team Head to Head Price: {row['team_head_to_head_odds_home']}
            Away Team: {row['team_away']},
            Away Team Position: {row['position_away']},
            Away Team Head to Head Price: {row['team_head_to_head_odds_away']}
            """
    
    # Generate input_picks string by iterating over 'tipper_picks' dataframe and formatting data into string
    input_picks = ""
    for index, row in tipper_picks.iterrows():
        input_picks += f"""
            Team: {row['team']},
            Price: {row['price']}
            """

    # Generate the prompt string to be used with the AI model
    prompt = f"""
        I have a set of predictions for NRL games in {predictions['round_name'].unique()[0]} {predictions['competition_year'].unique()[0]} made by a machine learning pipeline called the Footy Tipper: \n{input_predictions}\n 
        The description of the columns of interest is:
        
        * Home Team Result: the predicted result of the home team
        * Home Team: the home team
        * Home Team Position: the home team's position on the NRL ladder
        * Home Team Head to Head Price: the price bookies are offering for a home win
        * Away Team: the away team
        * Away Team Position: the away team's position on the NRL ladder
        * Away Team Head to Head Price: the price bookies are offering for an away win
        
        It also comes up with some good value tips for those interested in a punt in \n{input_picks}\n. If it is empty there isn't much value for punting in the round. The description of the columns of interest is:
        
        * Team = Team that is a good value pick
        * Price = what the bookies are offering them at
        
        Could you write up an email to my mates from Reg Reagan, giving them a synopsis of the round along with the tips? 
        Accompany the tips with some smart arsed comments about the teams playing.
        Remember to link everyone to the tips folder: {folder_url}
        Also, tell everyone to bring back the biff at the end of the email.
        Also also your favorite team is the Newcastle Knights.
        Always sign off the email as Reg Reagan.
        """

    # Use the AI model to generate the email content based on the prompt
    reg_regan = llm.predict(prompt)

    return reg_regan

# The 'send_emails' function sends an email to a list of recipients. The email details are prepared and the SMTP server is used to send the emails.
def send_emails(doc_name, subject, message, sender_email, sender_password, json_path):

    # 
    scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

    # Authorize Google client using service account credentials to access Google Sheets
    creds = service_account.Credentials.from_service_account_file(json_path, scopes=scope)
    client = gspread.authorize(creds)

    # Open the spreadsheet and get the data
    sheet = client.open(doc_name).sheet1 # this is the spreadsheet with the emails
    email_data = sheet.get_all_records()  # gets all the data inside your Google Sheet

    # Extract the recipient emails from the Google Sheet data
    recipient_emails = [row['Email'] for row in email_data]  # replace 'Email' with your actual column name

    # Prepare the email message using MIMEText
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ', '.join(recipient_emails)
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Setup the SMTP server for sending the email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    # Login to the SMTP server using sender's email and password
    server.login(sender_email, sender_password)

    # Send the email to the list of recipients
    text = msg.as_string()
    server.sendmail(sender_email, recipient_emails, text)

    # Close the SMTP server connection
    server.quit()
