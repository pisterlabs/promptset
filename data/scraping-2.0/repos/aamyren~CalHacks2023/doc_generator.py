import openai
import os
import json
import pprint
import re
from data_generator import DataGenerator

class DocGenerator:
    def __init__(self):
        pass
    
    # write the post in text format with timestamps 
    def write_data(self, filename):
        filename = filename.split('/')[1]
        # check if file exists
        check_file = os.path.isfile(filename[:-5]+'.txt')
        if check_file:
            return
        else:
            f = open(filename, 'r')
            json_obj = json.load(f)
            with open('generated_docs/'+filename[:-5]+'.txt', 'w') as f2:
                for post in json_obj:
                    # write with formatting
                    f2.write("POST: " + str([post["Time"]])+" ")
                    f2.write("POST TITLE: " + str(post["Title"]))
                    f2.write("\nURL : " + str(post["URL"]))
                    f2.write('\n')
                    f2.write("POST_TEXT: " + str(post["Text"]).strip('\n'))
                    f2.write('\n')
                    for comment in post["comments"]:
                        f2.write("\tCOMMENT: ")
                        f2.write(str(comment["timestamp"])+" ")
                        if comment["top_level"] != []:
                            f2.write(comment["top_level"][0].replace("\n", ""))
                            f2.write('\n')

        return None

    # helper function to get sentiment of text (optional, not implemented)
    def get_sentiment(text):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Sentiment analysis of the following text:\n{text}\n",
            temperature=0.5,
            max_tokens=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )

        sentiment = response.choices[0].text.strip()
        return sentiment


