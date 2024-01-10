import asyncio
import time
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import yaml
import httpx
from loguru import logger
import openai
from openai import OpenAIError
from config import OPENAI_API_KEY, OPENAI_COMPLETION_OPTIONS, PROXY_URL


class GptSpamClassifierValidation:
    def __init__(self, api_key: str = OPENAI_API_KEY, 
                 openai_completion_options: str = OPENAI_COMPLETION_OPTIONS, 
                 proxy: str = PROXY_URL,
                 timelimit: int = 5):
        """Initialize the classifier with an OpenAI API key."""
        self.api_key = api_key
        self.openai_completion_options = openai_completion_options
        self.timelimit = timelimit

        # If proxy is provided in .env file, it will be used when making requests to opeanai api
        http_client = httpx.AsyncClient(proxies=proxy) if proxy else None 
        self.client = openai.AsyncOpenAI(api_key=api_key, http_client=http_client)

        # Due to inability to get user_id when loading new data from Spam Monitoring channels, usernames are used instead in Validation model
        self.not_spam_ids = [] # list filles in when fitting the model

        with open("./prompts.yml", "r") as f:
            prompts = yaml.safe_load(f)
        
        self.prompt = prompts["spam_classification_prompt"]

        logger.info("Initialized GptClassifier")

    def fit(self, X: pd.DataFrame, y) -> None:
        """
        Fits the model to the training data.

        Parameters:
            X (pandas DataFrame): The input features of shape (n_samples, n_features).
            y (pandas DataFrame or numpy array): The target labels of shape (n_samples,).

        Returns:
            None
        """
        not_spam_mask = y == 0
        not_none_username_mask = X['from_id'].apply(lambda x: True if x is not np.NaN else False)
        self.not_spam_ids = X[(not_spam_mask) & (not_none_username_mask)]['from_id'].unique() # Using username column due to loaded data. Originally there are ids
        logger.info('The model was successfully fitted')
        return None

    async def predict(self, X: pd.DataFrame) -> List[dict]:
        """
        Predicts if the message is spam or not.

        Parameters:
            X (pandas DataFrame): The input data to predict spam/not-spam for.

        Returns:
            List[dict]: A list containing dictionaries with label, reasons for the answer, 
                        prompt_tokens, completion_tokens and time spent on request.
        """
        # Validate the input DataFrame to have the required columns
        logger.info("Predicting...")

        if not all(column in X for column in ['text', 'bio', 'from_id']):
            logger.error("Input DataFrame does not contain required columns: 'text', 'bio', 'from_id'.")
            return [{'label': None, 'reasons': "Input is missing required columns.", 'prompt_tokens': 0, 'completion_tokens': 0, 'time_spent': 0}]
         
        # Create a task for each row in the DataFrame
        tasks = [self._predict_row(X.iloc[i]) for i in range(len(X))]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        logger.info('Succesfully predicted')

        return results
    
    async def _predict_row(self, row) -> dict:
        start_time = time.time()
        time_spent = None

        text = row['text'][:600]
        bio = row['bio'][:100]

        prompt = self._create_prompt(text, bio)

        try:
            # Call the _api_call method with a timeout
            response = await asyncio.wait_for(self._api_call(prompt), timeout=self.timelimit)

            # Interpret the API response
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            response_text = response.choices[0].message.content

            label, reasons = self._process_response(response_text)
            time_spent = round(time.time() - start_time, 1)

            if label is None:
                logger.error("Couldn't interpret the OpenAI response")
                logger.debug(f"Response text: {response_text}")
                return {'label': None, 'reasons': "Couldn't interpret the OpenAI response", 'prompt_tokens': 0, 'completion_tokens': 0, 'time_spent': time_spent}
            
            logger.debug("Succesfully received response from OpenAI")
            return {'label': label, 'reasons': reasons, 'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'time_spent': time_spent}
        except asyncio.TimeoutError:
            # Handle the TimeoutError
            time_spent = round(time.time() - start_time, 1)
            logger.error("The OpenAI response took too long and was aborted after 5 seconds.")
            return {'label': None, 'reasons': 'Prediction timed out.', 'prompt_tokens': 0, 'completion_tokens': 0, 'time_spent': time_spent}
        except OpenAIError as e:
            # Handle OpenAI API errors
            time_spent = round(time.time() - start_time, 1) if time_spent is None else time_spent
            logger.exception("An error occurred with the OpenAI API: %s", e)
            return {'label': None, 'reasons': f'An OpenAI API error occurred: {e}', 'prompt_tokens': 0, 'completion_tokens': 0, 'time_spent': time_spent}
        except Exception as e:
            # Handle other unforeseen errors
            time_spent = round(time.time() - start_time, 1) if time_spent is None else time_spent
            logger.exception("An unexpected error occurred: %s", e)
            return {'label': None, 'reasons': f'An unexpected error occurred: {e}', 'prompt_tokens': 0, 'completion_tokens': 0, 'time_spent': time_spent}
        
    def _create_prompt(self, message: str, bio: str = None) -> str:
        """Create a prompt for the GPT model to classify the message."""
        prompt = self.prompt.format(message_text=message)
        return prompt
    
    async def _api_call(self, prompt: str):
        """Call the OpenAI API to get a response."""
        logger.info("Sending request to OpenAI API...")
        response = await self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[{"role": "user", "content": prompt}],
            **self.openai_completion_options
        )
        return response
    
    @staticmethod
    def _process_response(response: str) -> Tuple[Optional[int], Optional[str]]:
        """Process the response from OpenAI and extract the label and reasons."""
        lines = response.strip().split("\n")
        if not lines:
            return None, None
        
         # Determine the label based on the first line
        label = 1 if "<spam>" in lines[0] else 0 if "<not-spam>" in lines[0] else None

        # Join all lines except the first to form the reasons string; Empty if label == 0
        reasons = "\n".join(lines[1:]) if label == 1 else ''

        return label, reasons
