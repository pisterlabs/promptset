# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 Berkay Bozkurt <resitberkaybozkurt@gmail.com>
# SPDX-FileCopyrightText: 2023 Sophie Heasman <sophieheasmann@gmail.com>

import time
from collections import Counter

import numpy as np
import openai
import pandas as pd
import tiktoken
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from bdc.steps.helpers import TextAnalyzer
from bdc.steps.step import Step, StepError
from config import OPEN_AI_API_KEY
from database import get_database
from logger import get_logger

log = get_logger()


"""
HELPER FUNCTIONS
"""


def is_review_valid(review):
    """
    Checks if the review is valid (has text and original language).

    Parameters:
    review (dict): A dictionary representing a review.

    Returns:
    bool: True if the review is valid, False otherwise.
    """
    return not (review["text"] is None or review["lang"] is None)


def check_api_key(api_key, api_name):
    """
    Checks if an API key is provided for a specific API.

    Args:
        api_key (str): The API key to be checked.
        api_name (str): The name of the API.

    Raises:
        StepError: If the API key is not provided.

    Returns:
        bool: True if the API key is provided, False otherwise.
    """
    if api_key is None:
        raise StepError(f"An API key for {api_name} is needed to run this step!")
    else:
        return True


"""
CLASSES
"""


class GPTReviewSentimentAnalyzer(Step):
    """
    A class that performs sentiment analysis on reviews using GPT-4 model.

    Attributes:
        name (str): The name of the step.
        model (str): The GPT model to be used for sentiment analysis.
        model_encoding_name (str): The encoding name of the GPT model.
        MAX_PROMPT_TOKENS (int): The maximum number of tokens allowed for a prompt.
        no_answer (str): The default value for no answer.
        gpt_required_fields (dict): The required fields for GPT analysis.
        system_message_for_sentiment_analysis (str): The system message for sentiment analysis.
        user_message_for_sentiment_analysis (str): The user message for sentiment analysis.
        extracted_col_name (str): The name of the column to store the sentiment scores.
        added_cols (list): The list of additional columns to be added to the DataFrame.
        gpt (openai.OpenAI): The GPT instance for sentiment analysis.

    Methods:
        load_data(): Loads the GPT model.
        verify(): Verifies the validity of the API key and DataFrame.
        run(): Runs the sentiment analysis on the reviews.
        finish(): Finishes the sentiment analysis step.
        run_sentiment_analysis(place_id): Runs sentiment analysis on the reviews of a lead.
        gpt_sentiment_analyze_review(review_list): Calculates the sentiment score using GPT.
        extract_text_from_reviews(reviews_list): Extracts text from reviews and removes line characters.
        num_tokens_from_string(text): Returns the number of tokens in a text string.
        batch_reviews(reviews, max_tokens): Batches reviews into smaller batches based on token limit.
    """

    name = "GPT-Review-Sentiment-Analyzer"
    model = "gpt-4"
    model_encoding_name = "cl100k_base"
    text_analyzer = TextAnalyzer()
    MAX_PROMPT_TOKENS = 4096
    no_answer = "None"
    gpt_required_fields = {"place_id": "google_places_place_id"}
    system_message_for_sentiment_analysis = f"You are review sentiment analyzer, you being provided reviews of the companies. You analyze the review and come up with the score between range [-1, 1], if no reviews then just answer with '{no_answer}'"
    user_message_for_sentiment_analysis = "Sentiment analyze the reviews  and provide me a score between range [-1, 1]  : {}"
    extracted_col_name = "reviews_sentiment_score"
    added_cols = [extracted_col_name]
    required_cols = gpt_required_fields.values()
    gpt = None

    def load_data(self) -> None:
        """
        Loads the GPT model.
        """
        self.gpt = openai.OpenAI(api_key=OPEN_AI_API_KEY)

    def verify(self) -> bool:
        """
        Verifies the validity of the API key and DataFrame.

        Returns:
            bool: True if the API key and DataFrame are valid, False otherwise.
        """

        is_key_valid = check_api_key(OPEN_AI_API_KEY, "OpenAI")
        return super().verify() and is_key_valid

    def run(self) -> DataFrame:
        """
        Runs the sentiment analysis on the reviews.

        Returns:
            DataFrame: The DataFrame with the sentiment scores added.
        """
        tqdm.pandas(desc="Running sentiment analysis on reviews")
        self.df[self.extracted_col_name] = self.df[
            self.gpt_required_fields["place_id"]
        ].progress_apply(lambda place_id: self.run_sentiment_analysis(place_id))
        return self.df

    def finish(self) -> None:
        pass

    def run_sentiment_analysis(self, place_id):
        """
        Runs sentiment analysis on reviews of lead extracted from company's website.

        Args:
            place_id: The ID of the place.

        Returns:
            float: The average sentiment score of the reviews.
        """
        # if there is no reviews_path, then return without API call.
        if place_id is None or pd.isna(place_id):
            return None
        cached_result = get_database().fetch_gpt_result(place_id, self.name)
        if cached_result:
            return cached_result["result"]
        reviews = get_database().fetch_review(place_id)
        avg_score = self.textblob_calculate_avg_sentiment_score(reviews)
        get_database().save_gpt_result(avg_score, place_id, self.name)
        return avg_score

    def gpt_calculate_avg_sentiment_score(self, reviews):
        """
        Calculates the average sentiment score for a list of reviews using GPT.

        Args:
            reviews (list): A list of review texts.

        Returns:
            float: The average sentiment score.

        """
        review_texts = self.extract_text_from_reviews(reviews)
        # batch reviews so that we do not exceed the token limit of gpt4
        review_batches = self.batch_reviews(review_texts, self.MAX_PROMPT_TOKENS)
        scores = 0
        # iterate over each batch and calculate average sentiment score
        for review_batch in review_batches:
            sentiment_score = self.gpt_sentiment_analyze_review(review_batch)
            scores += sentiment_score or 0
        avg_score = scores / len(review_batches)
        return avg_score

    def textblob_calculate_avg_sentiment_score(self, reviews):
        """
        Calculates the average sentiment score for a list of reviews using TextBlob sentiment analysis.

        Args:
            reviews (list): A list of dictionaries containing review text and language information.
        Returns:
            float: The average sentiment score for the reviews.
        """
        reviews_langs = [
            {
                "text": review.get("text", ""),
                "lang": review.get("original_language", "en"),
            }
            for review in reviews
        ]
        if len(reviews_langs) == 0:
            return None
        scores = 0
        for review in reviews_langs:
            score = self.text_analyzer.calculate_sentiment_analysis_score(
                review["text"], review["lang"]
            )
            scores += score or 0
        avg_score = scores / len(reviews_langs)
        return avg_score

    def gpt_sentiment_analyze_review(self, review_list):
        """
        GPT calculates the sentiment score considering the reviews.

        Args:
            review_list: The list of reviews.

        Returns:
            float: The sentiment score calculated by GPT.
        """
        max_retries = 5  # Maximum number of retries
        retry_delay = 5  # Initial delay in seconds (5 seconds)

        for attempt in range(max_retries):
            try:
                log.info(f"Attempt {attempt+1} of {max_retries}")
                response = self.gpt.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_message_for_sentiment_analysis,
                        },
                        {
                            "role": "user",
                            "content": self.user_message_for_sentiment_analysis.format(
                                review_list
                            ),
                        },
                    ],
                    temperature=0,
                )
                # Extract and return the sentiment score
                sentiment_score = response.choices[0].message.content
                if sentiment_score and sentiment_score != self.no_answer:
                    return float(sentiment_score)
                else:
                    log.info("No valid sentiment score found in the response.")
                    return None
            except openai.RateLimitError as e:
                if attempt < max_retries - 1:
                    log.warning(
                        f"Rate limit exceeded, retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    log.error("Max retries reached. Unable to complete the request.")
                    break
            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.BadRequestError,
                openai.AuthenticationError,
                openai.PermissionDeniedError,
            ) as e:
                log.error(f"An error occurred with GPT API: {e}")
                break
            except Exception as e:
                log.error(f"An unexpected error occurred: {e}")
                break

        # Return None if the request could not be completed successfully
        return None

    def extract_text_from_reviews(self, reviews_list):
        """
        Extracts text from reviews and removes line characters.

        Args:
            reviews_list: The list of reviews.

        Returns:
            list: The list of formatted review texts.
        """
        reviews_texts = [review.get("text", None) for review in reviews_list]
        review_texts_formatted = [
            review.strip().replace("\n", " ") for review in reviews_texts if review
        ]
        return review_texts_formatted

    def num_tokens_from_string(self, text: str):
        """
        Returns the number of tokens in a text string.

        Args:
            text (str): The input text.

        Returns:
            int: The number of tokens in the text.
        """
        encoding = tiktoken.get_encoding(self.model_encoding_name)
        num_tokens = len(encoding.encode(text))
        return num_tokens

    def batch_reviews(self, reviews, max_tokens=4096):
        """
        Batches reviews into smaller batches based on token limit.

        Args:
            reviews: The list of reviews.
            max_tokens (int): The maximum number of tokens allowed for a batch.

        Returns:
            list: The list of batches.
        """
        batches = []
        current_batch = []
        current_count = self.num_tokens_from_string(
            self.user_message_for_sentiment_analysis
        )

        for review in reviews:
            token_count = self.num_tokens_from_string(review)
            if current_count + token_count > max_tokens:
                batches.append(current_batch)
                current_batch = [review]
                current_count = token_count
            else:
                current_batch.append(review)
                current_count += token_count

        if current_batch:
            batches.append(current_batch)

        return batches


class SmartReviewInsightsEnhancer(Step):
    """
    A step class that enhances review insights for smart review analysis.

    Attributes:
        name (str): The name of the step.
        required_fields (dict): A dictionary of required fields for the step.
        language_tools (dict): A dictionary of language tools for different languages.
        MIN_RATINGS_COUNT (int): The minimum number of ratings required to identify polarization.
        RATING_DOMINANCE_THRESHOLD (float): The threshold for high or low rating dominance in decimal.
        added_cols (list): A list of added columns for the enhanced review insights.

    Methods:
        load_data(): Loads the data for the step.
        verify(): Verifies if the required fields are present in the data.
        run(): Runs the step and enhances the review insights.
        finish(): Finishes the step.
        _get_language_tool(lang): Get the language tool for the specified language.
        _enhance_review_insights(lead): Enhances the review insights for a given lead.
        _analyze_rating_trend(rating_time): Analyzes the general trend of ratings over time.
        _quantify_polarization(ratings): Analyzes and quantifies the polarization in a list of ratings.
        _determine_polarization_type(polarization_score, highest_rating_ratio, lowest_rating_ratio, threshold): Determines the type of polarization based on rating ratios and a threshold.
        _calculate_average_grammatical_score(reviews): Calculates the average grammatical score for a list of reviews.
        _calculate_score(review): Calculates the score for a review.
        _grammatical_errors(text, lang): Calculates the number of grammatical errors in a text.

    """

    name = "Smart-Review-Insights-Enhancer"
    required_fields = {"place_id": "google_places_place_id"}
    text_analyzer = TextAnalyzer()
    MIN_RATINGS_COUNT = 1
    RATING_DOMINANCE_THRESHOLD = (
        0.4  # Threshold for high or low rating dominance in percentage (1.0 == 100%)
    )

    added_cols = [
        "review_avg_grammatical_score",
        "review_polarization_type",
        "review_polarization_score",
        "review_highest_rating_ratio",
        "review_lowest_rating_ratio",
        "review_rating_trend",
    ]

    def load_data(self) -> None:
        """
        Loads the data for the step.
        """
        pass

    def verify(self) -> bool:
        """
        Verifies if the required fields are present in the data.

        Returns:
            bool: True if the required fields are present, False otherwise.
        """
        return super().verify()

    def run(self) -> DataFrame:
        """
        Runs the step and enhances the review insights.

        Returns:
            DataFrame: The enhanced DataFrame with the added review insights.
        """
        tqdm.pandas(desc="Running reviews insights enhancement")

        # Apply the enhancement function
        self.df[self.added_cols] = self.df.progress_apply(
            lambda lead: pd.Series(self._enhance_review_insights(lead)), axis=1
        )
        return self.df

    def finish(self) -> None:
        """
        Finishes the step.
        """
        pass

    def _enhance_review_insights(self, lead):
        """
        Enhances the review insights for a given lead.

        Args:
            lead (pd.Series): The lead data.

        Returns:
            pd.Series: The enhanced review insights as a pandas Series.
        """
        place_id = lead["google_places_place_id"]
        if place_id is None or pd.isna(place_id):
            return pd.Series({f"{col}": None for col in self.added_cols})
        reviews = get_database().fetch_review(place_id)
        if not reviews:
            return pd.Series({f"{col}": None for col in self.added_cols})
        results = []
        reviews_langs = [
            {
                "text": review.get("text", ""),
                "lang": review.get("original_language", "en"),
            }
            for review in reviews
        ]
        avg_gram_sco = self._calculate_average_grammatical_score(reviews_langs)
        results.append(avg_gram_sco)

        ratings = [
            review["rating"]
            for review in reviews
            if "rating" in review and review["rating"] is not None
        ]

        polarization_results = list(self._quantify_polarization(ratings))
        results += polarization_results

        rating_time = [
            {
                "time": review.get("time"),
                "rating": review.get("rating"),
            }
            for review in reviews
        ]

        rating_trend = self._analyze_rating_trend(rating_time)
        results.append(rating_trend)

        extracted_features = dict(zip(self.added_cols, results))

        return pd.Series(extracted_features)

    def _analyze_rating_trend(self, rating_time):
        """
        Analyzes the general trend of ratings over time.

        Args:
            rating_time (list): List of review data, each a dict with 'time' (Unix timestamp) and 'rating'.

        Returns:
            float: A value between -1 and 1 indicating the trend of ratings.
                - A value close to 1 indicates a strong increasing trend.
                - A value close to -1 indicates a strong decreasing trend.
                - A value around 0 indicates no significant trend (stable ratings).
        """
        # Convert to DataFrame
        df = pd.DataFrame(rating_time)

        # Convert Unix timestamp to numerical value (e.g., days since the first review)
        df["date"] = pd.to_datetime(df["time"], unit="s")
        df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

        # Linear regression
        model = LinearRegression()
        model.fit(df[["days_since_start"]], df["rating"])

        # Slope of the regression line
        slope = model.coef_[0]

        # Normalize the slope to be within the range [-1, 1]
        slope_normalized = np.clip(slope, -1, 1)

        # Replace -0 with 0
        return 0 if slope_normalized == 0 else slope_normalized

    def _quantify_polarization(self, ratings: list):
        """
        Analyzes and quantifies the polarization in a list of ratings.

        Args:
            ratings (list): List of ratings.

        Returns:
            tuple: A tuple containing the polarization type, polarization score,
                highest rating ratio, and lowest rating ratio.
        """

        total_ratings = len(ratings)
        if total_ratings <= self.MIN_RATINGS_COUNT:
            log.info(f"There is no sufficient data to identify polarization")
            return "Insufficient data", None, None, None

        rating_counts = Counter(ratings)
        high_low_count = rating_counts.get(5, 0) + rating_counts.get(1, 0)
        high_low_ratio = high_low_count / total_ratings
        middle_ratio = (total_ratings - high_low_count) / total_ratings
        highest_rating_ratio = rating_counts.get(5, 0) / total_ratings
        lowest_rating_ratio = rating_counts.get(1, 0) / total_ratings
        polarization_score = high_low_ratio - middle_ratio

        polarization_type = self._determine_polarization_type(
            polarization_score,
            highest_rating_ratio,
            lowest_rating_ratio,
            self.RATING_DOMINANCE_THRESHOLD,
        )

        return (
            polarization_type,
            polarization_score,
            highest_rating_ratio,
            lowest_rating_ratio,
        )

    def _determine_polarization_type(
        self, polarization_score, highest_rating_ratio, lowest_rating_ratio, threshold
    ):
        """
        Determines the type of polarization based on rating ratios and a threshold.

        Args:
            polarization_score (float): The polarization score.
            highest_rating_ratio (float): The highest rating ratio.
            lowest_rating_ratio (float): The lowest rating ratio.
            threshold (float): The threshold for high or low rating dominance.

        Returns:
            str: The type of polarization.
        """
        if polarization_score > 0:
            if highest_rating_ratio > threshold:
                return "High-Rating Dominance"
            elif lowest_rating_ratio > threshold:
                return "Low-Rating Dominance"
            return "High-Low Polarization"
        return "Balanced"

    def _calculate_average_grammatical_score(self, reviews):
        """
        Calculates the average grammatical score for a list of reviews.

        Args:
            reviews (list): List of reviews.

        Returns:
            float: The average grammatical score.
        """
        scores = [
            self._calculate_score(review)
            for review in reviews
            if is_review_valid(review)
        ]
        valid_scores = [score for score in scores if score is not None]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0

    def _calculate_score(self, review):
        """
        Calculates the score for a review.

        Args:
            review (dict): The review data.

        Returns:
            float: The calculated score.
        """
        num_errors = self.text_analyzer.find_number_of_grammatical_errors(
            review["text"], review["lang"]
        )
        num_words = len(review["text"].split())
        if num_words == 0 or num_errors is None:
            return None
        return max(1 - (num_errors / num_words), 0)
