# Example of using openai codex.
# https://www.youtube.com/watch?v=Ru5fQZ714x8

import os
import openai
import pandas as pd

openai.api_key = "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"  

'''
print("What is your questions?")
user_response = input()

chat_response = openai.Completion.create(
                                    engine = "text-davinci-003",
                                    prompt = user_response,
                                    temperature = 0.5,  # controls randomness in the chat answer.  
                                    max_tokens = 100,
                                    )
                                    
print( chat_response.choices[0].text ) 
'''                        

# Create a sample dataframe
df = pd.DataFrame({'A': [1, 2, 3, 4],
                   'B': [5, 6, 7, 8],
                   'C': [9, 10, 11, 12]})

# Get statistics of the dataframe using openai plugin
stats = openai.stats(df)

# Print the statistics
print(stats)
