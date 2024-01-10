import logging
import nltk
import openai
#import tiktoken
import sys, time, itertools
import PyPDF2
import pandas as pd
import itertools
from website.models import ApiKey
from flask_login import current_user

nltk.download('punkt')

logging.basicConfig(level=logging.DEBUG,format=' %(asctime)s - %(levelname)s - %(message)s')
#logging.disable()


#breaks up emails into a list of strings
def split_email(email_string):
    return nltk.sent_tokenize(email_string)


#trunicates the prompt if neccessary
def truncate_prompt(prompt, max_characters=500):
    if len(prompt) <= max_characters:
        return prompt
    else:
        truncated_prompt = prompt[:max_characters] + '...'  # Add ellipsis to indicate truncation
        return truncated_prompt



def super_summary(content):
    time.sleep(1)  # wait one second
    logging.debug('Starting super_summary function')
    logging.debug('______________BREAKING UP CONTENT_______________')
    broken_content = split_email(content)
    logging.debug(f'BROKEN CONTENT: {broken_content}')
    logging.debug('______________END OF BROKEN CONTENT_____________')
    time.sleep(1)  # wait one second

    # Group sentences into chunks of 4
    logging.debug('Starting grouping content into chunks')
    grouped_content = [' '.join(x) for x in itertools.zip_longest(*[iter(broken_content)]*16, fillvalue='')]
    logging.debug(f'GROUPED CONTENT: {grouped_content}')
    time.sleep(1)  # wait one second

    # Summarizes every 4 sentences of the email
    logging.debug('Starting summarizing chunks')
    summary_counter = 0
    summaries_list = []

    for chunk in grouped_content:
        prompt = chunk

        for i in range(2):  # Retry the entire chunk up to 2 times
            try:
                # Try to summarize the chunk
                logging.debug('Trying to connect to GPT API')
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Your job is to shorten and summarize all information as brief and accurate as possible. Your answer must be shorter than the prompt given you"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=2000,
                    n=1,
                    stop=None,
                    temperature=0.5  # 0.5 is base temperature
                )
                logging.debug('MEASURING REQUEST SIZE AFTER PROCESS')

                if response['usage']:
                    response_size = response['usage']
                    logging.debug(f"REQUEST SIZE: {response_size}")

                response = response['choices'][0]['message']['content']
                summary_counter += 1
                logging.debug(f"SUMMARY # {summary_counter}")
                logging.debug(f"GPT API CONNECTION SUCCESSFUL: {response}")
                summaries_list.append(response)
                logging.debug(f'SUMMARIES LIST: {summaries_list}')
                time.sleep(1)  # wait one second

                # Check if the API response is empty or too long
                if not response or len(response) > 500:
                    logging.debug('API response is empty or too long, trying truncation method')
                    truncated_prompt = truncate_prompt(prompt, max_characters=2000)
                    summaries_list.append(truncated_prompt)
                    logging.debug(f'SUMMARIES LIST: {summaries_list}')
                
                break  # If the chunk is processed successfully, break the outer loop

            except openai.OpenAIError as e:
                logging.debug(f"GPT API ERROR: {e}")
                time.sleep(30)  # Wait 30 seconds before trying again

            except openai.error.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                time.sleep(10)  # pauses for 10 seconds then continues

            except openai.error.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}")
                time.sleep(10)  # pauses for 10 seconds then continues

            except openai.error.RateLimitError as e:
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(60)  # pauses for 60 seconds then continues

            except openai.error.Timeout as e:
                print(f"OpenAI API request timed out: {e}")
                time.sleep(10)  # pauses for 10 seconds then continues

            except openai.error.InvalidRequestError as e:
                print(f"Invalid request to OpenAI API: {e}")
                time.sleep(10)  # pauses for 10 seconds then continues

            except openai.error.AuthenticationError as e:
                print(f"Authentication error with OpenAI API: {e}")
                time.sleep(10)  # pauses for 10 seconds then continues

            except openai.error.ServiceUnavailableError as e:
                print(f"OpenAI API service unavailable: {e}")
                time.sleep(10) #pauses for 10 seconds then continues

    logging.debug('Exiting super_summary function')
    return summaries_list









#this is for condencing prompt

start_time = time.time()

#_____________MAIN_______________


#! for the chat bot specificaly
def gpt_bot_response(history, user_prompt):
    logging.debug('Starting gpt_bot_response function')
    time.sleep(1)  # wait one second
    message_history = history + [{"role": "user", "content": user_prompt}]
    for i in range(2):  # Retry up to 2 times
        try:
            logging.debug('Trying to connect to GPT API')
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages= message_history,
                max_tokens=2000,
                n=1,
                stop=None,
                temperature=0.5  # 0.5 is base temperature
            )
            message = response['choices'][0]['message']['content']
            history.append({"role": "assistant", "content": message})
            logging.debug('Message from GPT API received and added to history')
            time.sleep(1)  # wait one second
            return message, history

        except openai.OpenAIError as e:
            logging.debug(f"GPT API ERROR: {e}")
            time.sleep(10)  # Wait 10 seconds before trying again
        
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            time.sleep(30) #pauses for 30 seconds then continues

        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            time.sleep(10) #pauses for 10 seconds then continues

        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            time.sleep(45) #pauses for 60 seconds then continues

        except openai.error.Timeout as e:
            print(f"OpenAI API request timed out: {e}")
            time.sleep(10) #pauses for 10 seconds then continues

        except openai.error.InvalidRequestError as e:
            print(f"Invalid request to OpenAI API: {e}")
            time.sleep(10) #pauses for 10 seconds then continues

        except openai.error.AuthenticationError as e:
            print(f"Authentication error with OpenAI API: {e}")
            time.sleep(10) #pauses for 10 seconds then continues

        except openai.error.ServiceUnavailableError as e:
            print(f"OpenAI API service unavailable: {e}")
            time.sleep(10) #pauses for 10 seconds then continues
    logging.debug('Exiting gpt_bot_response function')



#! for emails specificaly
def gpt_summarize(prompt):
    logging.debug('Starting gpt_summarize function')
    time.sleep(1)  # wait one second
    for i in range(2):  # Retry up to 2 times
        try:
            logging.debug('Trying to connect to GPT API')
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Your job is to summarize information as brief and accurate as possible"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                n=1,
                stop=None,
                temperature=0.5  # 0.5 is base temperature
            )
            message = response['choices'][0]['message']['content']
            logging.debug('Message from GPT API received')
            time.sleep(1)  # wait one second
            return message

        except openai.OpenAIError as e:
            logging.debug(f"GPT API ERROR: {e}")
            time.sleep(10)  # Wait 10 seconds before trying again
        
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            time.sleep(30) #pauses for 30 seconds then continues

        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            time.sleep(10) #pauses for 10 seconds then continues

        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            time.sleep(45) #pauses for 60 seconds then continues

        except openai.error.Timeout as e:
            print(f"OpenAI API request timed out: {e}")
            time.sleep(10) #pauses for 10 seconds then continues

        except openai.error.InvalidRequestError as e:
            print(f"Invalid request to OpenAI API: {e}")
            time.sleep(10) #pauses for 10 seconds then continues

        except openai.error.AuthenticationError as e:
            print(f"Authentication error with OpenAI API: {e}")
            time.sleep(10) #pauses for 10 seconds then continues

        except openai.error.ServiceUnavailableError as e:
            print(f"OpenAI API service unavailable: {e}")
            time.sleep(10) #pauses for 10 seconds then continues

    logging.debug('Exiting gpt_summarize function without returning a message')
    time.sleep(1)  # wait one second
    return None




#! for emails too large for gpt_summarize specificaly
def gpt_response_upgraded(content):
    logging.debug('Starting gpt_response_upgraded function')
    time.sleep(1)  # wait one second

    #will continue shortening prompt until the api can send it
    prompt = content
    logging.debug('Trying to generate initial summary')
    summary = gpt_summarize(prompt)

    while summary is None:
        logging.debug("ABOUT TO ENTER super_summary FUNCTION")
        prompt = str(super_summary(prompt))
        logging.debug('_______________________________________super_summary FUNCTION FINISHED________________________________________')
        logging.debug('Trying to generate a new summary with the shortened prompt')
        summary = gpt_summarize(prompt)

    logging.debug(f"SUMMARY: {summary}")

    end_time = time.time()

    elapsed_time = end_time - start_time
    logging.debug(f'CONTENT LENGTH: {len(content)}')
    logging.debug(f'TIME TAKEN TO FINISH SUMMARY: {str(elapsed_time)} seconds')

    logging.debug('Exiting gpt_response_upgraded function')
    time.sleep(1)  # wait one second

    return summary





#! for working with pdf's
def pdf_response(filepath):
    # Open the PDF file in read-binary mode
    pdf_file = open(filepath, "rb")

    # Create a PDF file reader object
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)

    # Get the number of pages in the PDF
    num_pages = pdf_reader.numPages

    # Initialize an empty string to hold the text
    pdf_text = ""

    # Loop over each page and extract the text
    for page in range(num_pages):
        page_obj = pdf_reader.getPage(page)
        pdf_text += page_obj.extractText()  # Change here

    # Close the PDF file
    pdf_file.close()

    print(pdf_text)


    gpt_response_upgraded(pdf_text)


    # TEST: increacing chunk from 4 to 8
    # soldering pdf test
    # 16,832 tokens
    # 2023-06-28 11:15:55,348 - DEBUG - CONTENT LENGTH: 69662
    # 2023-06-28 11:15:55,348 - DEBUG - TIME TAKEN TO FINISH SUMMARY: 22 minutes and 28.477 seconds.

    # TEST 2: increacing chunk from 8 to 16
    # soldering pdf test
    # 16,832 tokens
    # 2023-06-28 11:35:38,179 - DEBUG - CONTENT LENGTH: 69662
    # 2023-06-28 11:35:38,179 - DEBUG - TIME TAKEN TO FINISH SUMMARY: 7 minutes and 3.216 seconds.

    # TEST 3: increacing chunk from 16 to 32
    # soldering pdf test
    # 2023-06-28 11:43:00,250 - DEBUG - CONTENT LENGTH: 69662
    # 2023-06-28 11:43:00,250 - DEBUG - TIME TAKEN TO FINISH SUMMARY: 3 minutes and 11.717 seconds.






#! for working with excel files
def excel_response(sheet):

    #working with excel files
    df = pd.read_excel(sheet)
    logging.debug(f"excel COLUMNS: {df.columns}")



    df = df.dropna() # drop rows with missing values

    #takes random sample of data
    sample_frac = 1
    df_sample = df.sample(frac=sample_frac)
    logging.debug(f"SAMPLE SELECTED: ")
    data_as_text = ""

    for index, row in df.iterrows():
        data_as_text += f"Row {index}:\n"
        for column_name in df.columns:
            data_as_text += f"{column_name}: {row[column_name]}.\n"


    logging.debug(f"data_as_text value: {data_as_text}")


    prompt = str(df.columns) + str(df_sample)

    logging.debug(f"FINAL PROMPT: {prompt}")

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": f"You are an assistant working with excel spreadsheet data. Your primary job is to find trends, patterns, and connections between this data and inform the user. Be very specific, You are to perform mathmatical calculations if neccessary. always assume the user is refering to this data sample, you are free to manipulate and change anything."},
            {"role": "user", "content": prompt},
        ]
    )

    print(response['choices'][0]['message']['content'])


#sheet = input('Enter File Name: ') # Financial Sample.xlsx
#excel_response(sheet)