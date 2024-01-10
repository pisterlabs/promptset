# content_extractor.py

import re
import os
from os import path
import json
import pandas as pd
import requests
import newspaper
from newspaper import Article
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import multiprocessing
import openai
import logging
import logging.handlers
from datetime import datetime
from signal import signal, SIGINT
import time

from dotenv import load_dotenv
load_dotenv()

from utils import handler, LOGGING_CONFIG

# Configure logging
# logging.basicConfig(level=logging.INFO)
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Timeout in seconds
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 60 # 120

# User agent for HTTP requests
USER_AGENT = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

# Output folder results
OUTPUT_FOLDER_PATH = "output"

# Override SSL verification settings
old_merge_environment_settings = requests.Session.merge_environment_settings
os.environ['REQUESTS_CA_BUNDLE'] = 'cacert.pem' #'NRCAN-Root-2019-B64.cer'

# Set OpenAI API key
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) # for openai=1.3.0
openai.api_key = os.getenv('OPENAI_API_KEY')

class ContentExtractor:
    def __init__(self, openai_model="gpt-3.5-turbo", openai_temp=0.8, openai_max_tokens=100):
        # Set OpenAI parameters
        self.openai_model = openai_model
        self.openai_temp = openai_temp
        self.openai_max_tokens = openai_max_tokens
        
        # Download stopwords and punkt if not already present
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        
        # Get the set of English stopwords
        self.stop_words = set(stopwords.words('english'))
        
    def read_data(self, fn, url_col_name="LinkURI", pub_date_col_name="PublishedDate"):
        """Reads data from a CSV file.

        Args:
            fn (str): File name.
            url_col_name (str, optional): Name of the column with URLs. Defaults to "LinkURI".
            pub_date_col_name (str, optional): Name of the column with article publication dates. Defaults to "PublishedDate".

        Returns:
            pd.DataFrame: Dataframe with read data.
        """
        try:
            # Try reading the CSV file with UTF-8 encoding
            df = pd.read_csv(fn, sep='|', encoding='utf-8')
        except UnicodeDecodeError:
            # If UnicodeDecodeError occurs, try reading the file with cp1252 encoding (Windows encoding)
            df = pd.read_csv(fn, sep='|', encoding='cp1252')

        # Additional error handling for missing URL column
        if url_col_name not in df.columns:
            raise ValueError(f"Column '{url_col_name}' not found in the CSV file. Please check the column name or provide a valid column name.")
        
        if pub_date_col_name not in df.columns:
            raise ValueError(f"Column '{pub_date_col_name}' not found in the CSV file. Please check the column name or provide a valid column name.")

        # Rename the specified URL column to "URL", and article published date column to "PublishedDate"
        df.rename(columns={url_col_name: "URL"}, inplace=True)
        df.rename(columns={pub_date_col_name: "PublishedDate"}, inplace=True)
        
        # Additional error handling for an empty dataframe
        if df.empty:
            raise ValueError("The CSV file is empty. Please provide a valid non-empty CSV file.")

        return df

    def clean_text(self, text):
        """Cleans text by removing HTML tags and special characters.

        Args:
            text (str): Input text.

        Returns:
            str: Cleaned text.
        """
        try:
            # Remove HTML tags from the text using regular expressions
            cleaned_text = re.sub(r'<[^>]+>', '', text)
            
            # Remove special characters, leaving only alphanumeric characters, commas, periods, and spaces,and French letters with accents
            cleaned_text = re.sub(r"""[^a-zA-Z0-9éàèùçâêîôûëïü.,!-:;()"'\s]""", '', cleaned_text)
            
            # Remove extra whitespaces by splitting and rejoining the text
            cleaned_text = ' '.join(cleaned_text.split())

            return cleaned_text
        except Exception as e:
            # Handle any unexpected errors and provide an error message
            raise ValueError(f"Error occurred while cleaning text: {str(e)}. Please check the input text and try again.")

    def remove_stopwords(self, text):
        """Removes stopwords from the text.

        Args:
            text (str): Input text.

        Returns:
            str: Text with stopwords removed.
        """
        try:
            # Tokenize the input text
            tokens = word_tokenize(text)

            # Remove stopwords from the list of tokens
            filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]

            # Join the filtered tokens to form the cleaned text
            filtered_text = " ".join(filtered_tokens)

            return filtered_text

        except Exception as e:
            # Handle any unexpected errors
            print(f"An error occurred while removing stopwords: {e}")
            return text  # Return the original text in case of an error

    def extract_url_content(self, url, language='en'):
        """Extracts content from a given URL.

        Args:
            url (str): URL to extract content from.
            language (str, optional): Language of the content. Defaults to 'en'.

        Returns:
            tuple: Summary, content, and validity flag (1 if valid body, 0 otherwise).
        """
        signal(SIGINT, handler)
        
        summary, content, is_valid = '', '', -1

        try:
            # Make a request to the URL with error handling for SSL, timeout, and connection errors
            response = self.make_request(url)
        except Exception as e:
            logging.error(f"An error occurred during the request: {str(e)}")
            return summary, content, is_valid

        try:
            # Use newspaper library to extract content from the HTML
            article = self.extract_article(response, language)
            logging.info(f"{url}: content extracted")
            
            # Process and clean the extracted content
            summary = self.clean_text(article.summary)
            content = self.clean_text(article.text)
            is_valid = 1 if article.is_valid_body() else 0
        except Exception as e:
            logging.error(f"An error occurred during content extraction: {str(e)}")

        return summary, content, is_valid

    def make_request(self, url):
        """Makes a request to the given URL with error handling.

        Args:
            url (str): URL to make a request to.

        Returns:
            requests.Response: Response object.
        """
        try:
            response = requests.get(url, verify=True, headers=USER_AGENT, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        except requests.exceptions.SSLError:
            response = requests.get(url, verify=False, headers=USER_AGENT, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        except requests.exceptions.ReadTimeout:
            logging.error(f"Access to {url} timed out")
            raise  # Re-raise the exception to be caught in the higher level
        except requests.exceptions.ConnectionError:
            logging.error(f"Access to {url} refused")
            raise  # Re-raise the exception to be caught in the higher level

        return response

    def extract_article(self, response, language):
        """Extracts article information using the newspaper library.

        Args:
            response (requests.Response): Response object from the request.
            language (str): Language of the article.

        Returns:
            newspaper.Article: Article object.
        """
        article = newspaper.Article(url='', language=language)
        article.download(input_html=response.content)
        article.parse()
        article.nlp()

        return article

    def extract_content(self, df, num_processes=None, out_fn=None):
        """Extracts content in parallel from URLs in the dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing URLs.
            num_processes (int, optional): Number of processes for parallel extraction. Defaults to None.
            out_fn (str, optional): Output file name to save the results. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe with extracted content.
        """
        try:
            if num_processes is None:
                num_processes = multiprocessing.cpu_count() - 1
                
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.starmap(self.extract_url_content, zip(df['URL'], df['Language']))  

        except KeyboardInterrupt:
            logging.error('Got ^C while pool mapping, terminating the pool')
            pool.terminate()
            logging.error('Pool is terminated')
            logging.error('Joining pool processes')
            pool.join()
            logging.error('Join complete')

        df['Summary'], df['New_Content'], df['Is_Article'] = zip(*results)

        try:
            logging.info("Saving results ...")
            if out_fn is not None and out_fn != "":
                df.to_csv(out_fn, index=False, sep='|')
            else:
                if not os.path.exists(OUTPUT_FOLDER_PATH): 
                    os.makedirs(OUTPUT_FOLDER_PATH)
                
                # Get the current date and time    
                current_datetime = datetime.now().strftime('%Y-%m-%d_%H%M%S') 
                out_fn = f"extracted_url_content_{current_datetime}.csv"
                out_fn = os.path.join(OUTPUT_FOLDER_PATH, out_fn)
                df.to_csv(out_fn, index=False, sep='|')
            logging.info("Saved.")
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            logging.error("Data aren't saved but returned")
            pass

        return df

    def filter_scraped_data(self, df):
        """Filters the dataframe to include only valid articles.

        Args:
            df (pd.DataFrame): Dataframe with extracted content.

        Returns:
            pd.DataFrame: Filtered dataframe with valid articles.
        """
        try:
            # Check if 'Is_Article' column exists in the dataframe
            if 'Is_Article' not in df.columns:
                raise ValueError("Column 'Is_Article' not found in the dataframe. Please check the column name.")
            
            # Filter the dataframe to include only valid articles
            df_filtered = df[df["Is_Article"] == 1].reset_index(drop=True)
            
            return df_filtered

        except Exception as e:
            # Handle any unexpected errors and print a helpful message
            print(f"An error occurred during data filtering: {str(e)}")
            return pd.DataFrame()  # Return an empty DataFrame in case of an error

    def check_and_append_columns(self, df, ncol = 6):
        """
        Check the number of columns in a pandas DataFrame and append new columns if necessary.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            ncol (int): The minimum number of columns desired in the DataFrame.

        Returns:
            pd.DataFrame: The modified DataFrame with ncol columns.
        """
        current_columns = len(df.columns)

        if current_columns < ncol:
            # Determine the number of new columns needed
            num_new_columns = ncol - current_columns

            # Append new columns with default values
            for i in range(num_new_columns):
                new_column_name = f'NewColumn_{i + 1}'
                df[new_column_name] = '' 

        return df

    def extract_single_event_chatopenai(self, url_content, url, language, publish_date):
        """Extracts information for a single event using OpenAI API.

        Args:
            url_content (str): Content of the URL.
            url (str): URL of the event.
            language (str): Language of the content ('en' for English, 'fr' for French).
            publish_date (str): Date of the URL article publication.

        Returns:
            pd.DataFrame: Dataframe with extracted information.
        """
        try:
            logger.info(f"OpenAI is extracting information from {url}")

            # Define system and user messages based on the language
            system_msg, user_msg = self.prepare_messages(language, url_content)

            # Make an OpenAI API call
            openai_content = self.make_openai_call(system_msg, user_msg)

            # Transform OpenAI response to DataFrame
            content_df = self.transform_openai_response_to_df(openai_content)

            # Add metadata to the DataFrame
            content_df["link"] = url
            content_df["published_date"] = publish_date

            # Pause for 60 seconds to avoid API rate limits
            time.sleep(60)

            return  content_df # openai_content #

        except Exception as e:
            # Handle any unexpected errors
            logger.error(f"An error occurred during extraction: {str(e)}")
            return pd.DataFrame()  # Return an empty DataFrame in case of an error

    def prepare_messages(self, language, url_content):
        """Prepare system and user messages based on the language.

        Args:
            language (str): Language code ('en' or 'fr').
            url_content (str): Context from the URL.

        Returns:
            tuple: System message and user message.
        """
        if language == 'en':
            system_msg = "You are a helpful assistant. You answer all the questions. Your responses consist of valid JSON syntax, with no other comments, explanations, reasoning, or dialogue that do not consist of valid JSON. Each key is the question number. You do not include the questions themselves. Each value is the corresponding answer. Each key-value pair should be enclosed in curly braces, and each key and value should be enclosed in double-quotes."            # system_msg = "You are a helpful assistant. You answer all the questions. Your responses consist of valid JSON syntax. Do not include any additional comments, explanations, reasoning, or dialogue not consisting of valid JSON. Include the number of question only. Do not include any questions. Each question number and answer should be enclosed in double-quotes. "
            # system_msg = 'You are a helpful assistant. You answer all the questions. Your responses consist of valid JSON syntax, with no other comments, explanations, reasoning, or dialogue not consisting of valid JSON. You put each answer '
            quest1 = "1. Did a flood event occur? (Respond with 'Yes' or No' only. If the answer is 'No', mark the following questions as 'NA'.)"
            quest2 = "2. If a flood event occurred, what caused the flood event? (Specify the cause or mark as Unknown)"
            # quest3 = "3. If a flood event happened and its cause is known, what is the name of this cause? (Name of the cause only or Unknown)"
            quest3 = "3. If a flood event occurred, when did it happen? (Specify in YYYY-MM format or mark as Unknown)"
            quest4 = "4. If a flood event occurred, where did it happen? (Specify all affected places or mark as Unknown)"
            quest5 = "5. Did any casualties occur if a flood event took place? (Yes or No or mark as Unknown)"
            quest6 = "6. Did evacuation take place if a flood event occurred? (Yes or No or mark as Unknown)"
            quest7 = "7. If the locations of the flood-affected areas are known, specify the country they are in? (or mark as Unknown)"
            
            user_msg = f"Questions answering: \nContext: {url_content}\n {quest1}\n {quest2}\n {quest3}\n {quest4}\n {quest5}\n {quest6}\n {quest7}"            
        elif language == 'fr':
            system_msg = "Vous êtes un assistant fournissant des réponses utiles. Vous répondez à toutes les questions. Vos réponses consistent en une syntaxe JSON valide, sans autres commentaires, explications, raisonnements ou dialogues qui ne sont pas constitués de syntaxe JSON valide. Chaque clé est le numéro de la question. N'incluez pas les questions elles-mêmes. Chaque valeur est la réponse correspondante. Chaque paire clé-valeur doit être enfermée dans des accolades, et chaque clé et valeur doivent être enfermées entre guillemets."
            quest1 = "1. Est-ce qu'un événement d'inondation s'est produit ? (Oui ou Non seulement. Si la réponse est 'Non', marquez les questions suivantes comme 'NA'.)"
            quest2 = "2. Si un événement d'inondation s'est produit, quelle en était la cause ? (Spécifiez la cause ou marquez comme Inconnu)"
            quest3 = "3. Si un événement d'inondation s'est produit, quand s'est-il produit ? (Spécifiez au format AAAA-MM ou marquez comme Inconnu)"
            quest4 = "4. Si un événement d'inondation s'est produit, où s'est-il produit ? (Spécifiez tous les endroits affectés ou marquez comme Inconnu))"
            quest5 = "5. Y a-t-il eu des victimes en cas d'inondation? (Oui ou Non ou marquer comme Inconnu)"
            quest6 = "6. Est-ce qu'une évacuation a eu lieu en cas d'inondation ? (Oui, Non ou marquer comme Inconnu)"
            quest7 = "7. Si les emplacements des zones touchées par l'inondation sont connus, spécifiez le pays dans lequel ils se trouvent? (ou marquez comme Inconnu)"
            
            user_msg = f"Réponses aux questions : \nContexte: {url_content}\n {quest1}\n {quest2}\n {quest3}\n {quest4}\n {quest5}\n {quest6}\n {quest7}"          
        else:
            logging.error("The provided mode is not recognized.")
            raise ValueError("The provided mode is not recognized.")

        return system_msg, user_msg

    def make_openai_call(self, system_msg, user_msg):
        """Make an OpenAI API call based on the chosen model.

        Args:
            system_msg (str): System message for OpenAI.
            user_msg (str): User message for OpenAI.

        Returns:
            str: OpenAI response content.
        """
        try:
            if self.openai_model in ["gpt-3.5-turbo", "gpt-3.5-turbo-1106"]:
                response = openai.ChatCompletion.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    max_tokens=self.openai_max_tokens,
                    temperature=self.openai_temp
                )
                openai_content = response["choices"][0]["message"]["content"]
            else:
                prompt = system_msg + '\n' + user_msg
                response = openai.Completion.create(
                    model=self.openai_model,
                    prompt=prompt,
                    max_tokens=self.openai_max_tokens,
                    temperature=self.openai_temp
                )
                openai_content = response['choices'][0]['text']

            return openai_content

        except Exception as e:
            # Handle any unexpected errors during the OpenAI API call
            logger.error(f"An error occurred during the OpenAI API call: {str(e)}")
            raise
    
    def transform_openai_response_to_df(self, openai_content):
        """Transforms OpenAI response into a dataframe.

        Args:
            openai_content (str): OpenAI response content.

        Returns:
            pd.DataFrame: Dataframe with transformed OpenAI content.
        """
        try:
            if self.openai_model in ["gpt-3.5-turbo", "gpt-3.5-turbo-1106"]:
                # Replace newline characters and leading/trailing spaces
                openai_content = ''.join(char for char in openai_content if char.isprintable())
            else:
                # Process OpenAI response for other models
                pattern = re.compile(r'Answers(\d+): (.+)')
                matches = pattern.findall(openai_content)
                data_dict = {f'Answers{index}': value for index, value in matches}
                openai_content = json.dumps(data_dict, indent=4)
                openai_content = re.sub("Answers", "Question", openai_content)
                openai_content = openai_content.replace('[', '').replace(']', '').replace('\n ', '')
                openai_content = ''.join(char for char in openai_content if char.isprintable())

            openai_content_dict = json.loads(openai_content)
            openai_content_df = pd.DataFrame([openai_content_dict])

            # Define column names
            column_names = ["is_happened", "flood_cause_en", "date", "location", "death", "evacuation", "country"]

            # Check and append columns
            openai_content_df = self.check_and_append_columns(openai_content_df, ncol=len(column_names))
            openai_content_df.columns = column_names
        
            return openai_content_df

        except Exception as e:
            # Handle any unexpected errors and print a helpful message
            logger.error(f"An error occurred during transformation: {str(e)}")

            # Provide a default structure in case of an error
            json_string = {'is_happened': openai_content}
            openai_content_df = pd.DataFrame(json_string)

            # Define column names
            column_names = ["is_happened", "flood_cause_en", "date", "location", "death", "evacuation", "country"]

            # Check and append columns for default structure
            openai_content_df = self.check_and_append_columns(openai_content_df, ncol=len(column_names))
            openai_content_df.columns = column_names

            return openai_content_df

    def extract_events_chatopenai(self, df, num_processes=None, out_fn=None): #, openai_model="gpt-3.5-turbo", openai_temp=0.8, openai_max_tokens=150, out_fn=None):
        """Extracts information for multiple events using OpenAI API.

        Args:
            df (pd.DataFrame): Dataframe with content and URLs.
            num_processes (int, optional): Number of processes for parallel extraction. Defaults to None.
            out_fn (str, optional): Output file name to save the results. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe with extracted information for multiple events.
        """
        try:
            # Set the default number of processes if not provided
            if num_processes is None:
                num_processes = multiprocessing.cpu_count() - 1

            # Prepare arguments for parallel processing
            model_args = [self.openai_model] * df.shape[0]
            temp_args = [self.openai_temp] * df.shape[0]
            tokens_args = [self.openai_max_tokens] * df.shape[0]

            # Use multiprocessing for parallel extraction
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.starmap(
                    self.extract_single_event_chatopenai,
                    zip(df['New_Content'], df['URL'], df['Language'], df['PublishedDate'])
                )

            # Combine results into a single DataFrame
            results_df = pd.concat(results, axis=0)

        except KeyboardInterrupt:
            # Handle KeyboardInterrupt to terminate the pool gracefully
            logger.error('Got ^C while pool mapping, terminating the pool')
            pool.terminate()
            logger.error('Pool is terminated')
            logger.error('Joining pool processes')
            pool.join()
            logger.error('Join complete')

        try:
            # Save results to a CSV file if an output filename is provided
            logging.info("Saving results ...")
            if out_fn is not None and out_fn != "":
                results_df.to_csv(out_fn, index=False, sep='|')
            else:
                if not os.path.exists(OUTPUT_FOLDER_PATH): 
                    os.makedirs(OUTPUT_FOLDER_PATH)
                
                # Get the current date and time    
                current_datetime = datetime.now().strftime('%Y-%m-%d_%H%M%S') 
                out_fn = f"openai_results_{current_datetime}.csv"
                out_fn = os.path.join(OUTPUT_FOLDER_PATH, out_fn)
                results_df.to_csv(out_fn, index=False, sep='|')
            logging.info("Saved.")

        except Exception as e:
            # Handle exceptions during the saving process
            logger.error(f"An error occurred while saving results: {str(e)}")

        # Return the DataFrame with extracted information
        return results_df