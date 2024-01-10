from twitch_listener import utils
import argparse
import json
import sqlite3
import pandas as pd
import numpy as np
import openai


class dataPreparation:
    def __init__(self):
        self.chat_df = pd.DataFrame()
        self.OPENAI_API_KEY = ""
        with open('config.json', 'r') as file_to_read:
            json_data = json.load(file_to_read)
            self.OPENAI_API_KEY = json_data["OPENAI_API_KEY"]
        
    def getMaxDate(self):
    # ************************************************************
    # Get the max date from the chat table
    # *************************************************************
        # Create a SQL connection to our SQLite database
        try:
            conn = sqlite3.connect('../data/db.sqlite3',isolation_level=None)
            cur = conn.cursor()
            query = '''SELECT DISTINCT MAX(stream_datetime) FROM chats'''
                    
            cur.execute(query) 
            result = cur.fetchall() 
            maxDate = result[0][0]
            
            cur.close()
            conn.close()
            return maxDate
        
        except sqlite3.Error as error:
            print("Failed to read data from chat table", error) 
            
        finally:
            if conn:
                conn.close()
            

    def selectNewChats(self):
    # ************************************************************
    # Select new chats from the chat table and load them into a dataframe.
    # ************************************************************    
        maxDate = self.getMaxDate()
        
        try:
            # Create a SQL connection to our SQLite database
            conn = sqlite3.connect('../data/db.sqlite3',isolation_level=None)
            cur = conn.cursor() 
            select_query = '''SELECT * FROM chats_table_demo WHERE stream_datetime > '%s' ''' % (maxDate)
            
            # Load the data into a DataFrame
            self.chat_df = pd.read_sql(select_query, conn)
            print(self.chat_df.head())
            print("***************************************************************************")
            print("number of new chats to prep and load into chat table:- ",self.chat_df.shape[0])
            print("***************************************************************************")
           
            # Be sure to close the connection
            cur.close()
            conn.close()   
        
        except sqlite3.Error as error:
            print("Failed to read new chats from db table", error) 
            
        finally:
            if conn:
                conn.close()    

        
    def emote_lookup(self):
    # ************************************************************
    #  Read Emote fact file , convert it to json and save it as emotes.json
    # ************************************************************
        csvFilePath = '../data/facttable/EmoteFactTable.csv'
        jsonFilePath = '../data/emotes.json' 
        
        utils.emote_to_json(csvFilePath,jsonFilePath)
        emote_json = utils.reload_json(jsonFilePath)
    
        self.chat_df['new_message_text'] = self.chat_df['message_text'].map(lambda x: utils.replace_emoticons(x, emote_json))

    def data_clean(self):  
    # ************************************************************
    # Clean the new chats before loading into the database.
    # ************************************************************ 
        # Create a new column with the same data but without the extra spaces
        #self.chat_df['new_message_text']  = self.chat_df['new_message_text'] .str.replace('\s+', ' ')
        
        # Remove new line characters
        self.chat_df['new_message_text'] = self.chat_df['new_message_text'].str.replace('\n', ' ')  
    
        # Remove special characters
        self.chat_df['new_message_text'] = self.chat_df['new_message_text'].str.replace("[^a-zA-Z]", " ")
        self.chat_df['general_sentiment'] =''
        self.chat_df['specific_sentiment'] =''
        
    def loadIntoDatabase(self):
    # ************************************************************
    # Load the new chats into the database
    # ************************************************************
        
        try:
            # Create a SQL connection to our SQLite database
            #conn = sqlite3.connect('../data/chat_table.sqlite3',isolation_level=None)
            conn = sqlite3.connect('../data/db.sqlite3',isolation_level=None)
            # Create a cursor object
            cur = conn.cursor()
            
            # Create the insert statement for the data
            insert_stmt = '''INSERT INTO chats(
                                            date, 
                                            stream_datetime,
                                            stream_length,
                                            username, 
                                            message_text, 
                                            channel_name,
                                            stream_topic,
                                            stream_title,
                                            chatter_count,
                                            viewer_count,
                                            follower_count,
                                            subscriber_count,
                                            stream_date,
                                            stream_id,
                                            message_sentiment,
                                            new_message_text,
                                            general_sentiment,
                                            specific_sentiment) 
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''         
            
            # Execute the insert statement for each row of data
            for row in self.chat_df.itertuples():
                #print(row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18])
                cur.execute(insert_stmt, (row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18]))  
            
            #self.chat_df.to_sql('chats', conn, index = False)
            #Commit the change
            conn.commit()
            # Be sure to close the connection
            cur.close()
            conn.close()
            
        except sqlite3.Error as error:
                print("Failed to insert into chat table", error) 
        
        finally:
            if conn:
                conn.close()     
        
        
    def main(self):
        self.selectNewChats()
        self.emote_lookup()
        self.data_clean()
        self.loadIntoDatabase()
        
if __name__ == "__main__":
    dataPreparation = dataPreparation()
    dataPreparation.main()
    