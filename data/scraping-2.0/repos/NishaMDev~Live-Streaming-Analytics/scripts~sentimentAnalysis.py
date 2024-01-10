##5 * * * * /home/aspera/my_script.sh

from scripts.twitch_listener import utils
import argparse
import json
import sqlite3
import pandas as pd
import numpy as np
import openai


class sentimentAnalyzer:
    def __init__(self):
        self.chat_df = pd.DataFrame()
        self.OPENAI_API_KEY = ""
        
        with open('config.json', 'r') as file_to_read:
            json_data = json.load(file_to_read)
            self.OPENAI_API_KEY = json_data["OPENAI_API_KEY"]
        
    def processSentiment(self):
    # ************************************************************
    # READ THE CHAT TABLE AND PROCESS THE SENTIMENT.
    # ************************************************************
        try:
            #conn = sqlite3.connect('../data/chat_table.sqlite3',isolation_level=None)
            conn = sqlite3.connect('../data/db.sqlite3',isolation_level=None)
            cur = conn.cursor()
            
            offset,limit = 0 ,100  
            iteration = 0  
            
            while True:
                query = '''SELECT date,
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
                                new_message_text,
                                general_sentiment,
                                specific_sentiment
                            FROM chats 
                            WHERE general_sentiment = ''
                            LIMIT %d OFFSET %d;''' % (limit, offset)
                
                cur.execute(query) 
                result = cur.fetchall()
                
                iteration = iteration+1     
                print("********************************************")   
                print("Batch Iteration--> "+ str(iteration))
                print("Length of result",len(result))  
                #print("Result",result)
                         
                if len(result) != 0:
                #if iteration == 1 or  len(result) == 0:
                    chat_batch_str = self.build_chat_batch(result)
                    response_genlist = self.get_sentiment_score(chat_batch_str)
                    self.updateSentimentInTable(conn, result, response_genlist)                      
                else:  
                    print("No more rows to process")
                    break   
                
                offset += limit
                
            # Be sure to close the connection
            cur.close()
            conn.close() 
            
        except sqlite3.Error as error:
            print("Failed to read data from chat table for sentiment analysis", error) 
       
        finally:
            if conn:
                # Be sure to close the connection
                conn.close() 

    def updateSentimentInTable(self, conn, result, response_genlist):
        try:
            i=0
            update_cur = conn.cursor()
            print("Length of response_genlist",len(response_genlist))
            #print("response_genlist",response_genlist)
            for rows in result:
                #print("update row value:-",response_genlist[i] ,rows[0],rows[1],rows[3],rows[14])
                update_query = '''UPDATE chats SET  general_sentiment = '%s'
                                      WHERE date = '%s'
                                        AND stream_datetime = '%s'
                                        AND username = '%s'
                                        AND new_message_text = '%s'
                                        ;''' % (response_genlist[i], rows[0],rows[1],rows[3],rows[14])             
                update_cur.execute(update_query)                   
                i=i+1
            print("Number of rows updated",i)
        
        except sqlite3.Error as error:
            print("Failed to update sentiment score for \
                                        date - {rows[0]},stream_datetime - {rows[1]},\
                                        username - {rows[3]} and new_message_text - {rows[14]}", error)     

    def get_sentiment_score(self, chat_batch_str):
        openai.api_key =  self.OPENAI_API_KEY
        response_genlist = utils.general_sentiments(chat_batch_str)
        return response_genlist
    
    def get_sentiment_probability(self, chat_str):
        openai.api_key =  self.OPENAI_API_KEY
        response_senti_probability = utils.sentiment_probability(chat_str)
        return response_senti_probability

    def build_chat_batch(self, result):
        chat_iter = 1
        chat_batch_str=''
        for rows in result:
            chat = rows[14]
            chat_batch_str = chat_batch_str + str(chat_iter) +'.'+'"'+ chat +'"'+'\n'
            chat_iter=chat_iter+1
        return chat_batch_str   
        
        
    def main(self):
        self.processSentiment()     

        
if __name__ == "__main__":
    sentimentAnalyzer = sentimentAnalyzer()
    sentimentAnalyzer.main()
    