import os
import openai
import time
import csv
import json
import pandas as pd
import logging
from openai import OpenAI
import httpx
from typing import List, Dict, Optional
import scalexi.utilities.data_formatter as dfm
from scalexi.document_loaders.context_loaders import ContextExtractor

# Read logging level from environment variable
logging_level = os.getenv('LOGGING_LEVEL', 'WARNING').upper()

# Configure logging with the level from the environment variable
logging.basicConfig(
    level=getattr(logging, logging_level, logging.WARNING),  # Default to WARNING if invalid level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a logger object
logger = logging.getLogger(__name__)

RETRY_SLEEP_SECONDS = 0.5

DEFAULT_SYSTEM_PROMPT = """You are an assistant to create a JSON Array of prompt and completions from a context. 
                                Return prompt and completion as a JSON ARRAY structure:\n"
                                [{"prompt": "question1", "completion": "answer1"},{"prompt": "question2", "completion": "answer2"}]"
                                """

class PromptCompletionGenerator:
    """
    A class for generating prompt completions using the OpenAI API.

    This class is designed to interact with the OpenAI API to generate responses based on provided prompts. 
    It supports custom timeout settings and handles the API interactions to fetch prompt completions.

    :method __init__: Initialize the PromptCompletionGenerator instance.
    :type __init__: constructor

    :param openai_key: An optional string representing the OpenAI API key. If not provided, the key is fetched from the environment variable "OPENAI_API_KEY".
    :type openai_key: Optional[str], default=None
    :param enable_timeouts: A flag to enable custom timeout settings for the OpenAI client.
    :type enable_timeouts: bool, default=False
    :param timeouts_options: A dictionary specifying the timeout settings. It is only used if enable_timeouts is set to True.
    :type timeouts_options: Optional[dict], default=None

    :return: An instance of PromptCompletionGenerator.
    :rtype: PromptCompletionGenerator

    :raises ValueError: Raised if the provided or fetched OpenAI API key is invalid.

    :example:

    ::

        >>> generator = PromptCompletionGenerator(openai_key="sk-xxxxx")
        >>> print(type(generator))
        <class 'PromptCompletionGenerator'>
    """

    def __init__(self, openai_key: Optional[str] = None, enable_timeouts= False, timeouts_options= None):
        # Set the OpenAI API key
        self.openai_api_key = openai_key if openai_key is not None else os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key or not self.openai_api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key.")
        self.client = OpenAI(api_key=self.openai_api_key, max_retries=3)
        if enable_timeouts:
            if timeouts_options is None:
                timeouts_options = {"total": 120, "read": 60.0, "write": 60.0, "connect": 10.0}
                self.client = self.client.with_options(timeout=httpx.Timeout(120.0, read=60.0, write=60.0, connect=10.0))
            else:
                self.client = self.client.with_options(timeout=httpx.Timeout(timeouts_options["total"], timeouts_options["read"], timeouts_options["write"], timeouts_options["connect"]))
        self.context_extractor = ContextExtractor()

        # Set the API key for the OpenAI client
        openai.api_key = self.openai_api_key

        # Set the default retry sleep seconds
        self.retry_sleep_seconds = 0.5

        # Set the default system prompt
        self.default_system_prompt = """You are an assistant to create prompt and completions from a context. 
                                        Return prompt and completion as a JSON ARRAY structure:\n"
                                        [{"prompt": "question1", "completion": "answer1"},
                                         {"prompt": "question2", "completion": "answer2"}]"""
        
        self.data_formatter = dfm.DataFormatter()


    def parse_and_save_json(self, json_string, output_file, context=None):
        """
        Parses a JSON-formatted string and persists it to a file with optional context.

        This function parses a given JSON-formatted string into structured data and saves it into a JSON file. 
        It allows the inclusion of additional contextual data, enriching the content when provided.

        :param json_string: A string formatted in JSON, representing structured data to be parsed.
        :type json_string: str
        :param output_file: The destination file path where the parsed JSON data will be saved.
        :type output_file: str
        :param context: Optional context to be added to the parsed data, augmenting the information.
        :type context: str, optional

        :note: The `context` should align with the structure of the JSON string for consistency in the output file.

        :example:

        ::

            >>> json_str = '[{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]'
            >>> output_path = 'output.json'
            >>> context_info = 'Additional context information.'
            >>> parse_and_save_json(json_str, output_path, context_info)
            # This will parse the JSON string and save it to 'output.json' with additional context if provided.
        """

        try:
            # Load the JSON array into a list of dictionaries
            data_list = json.loads(json_string)

            # Transform the list of dictionaries into a pandas DataFrame
            df = pd.DataFrame(data_list)

            # Initialize a DataFrame to hold the newly formatted data
            formatted_data = pd.DataFrame(columns=['formatted_prompt', 'formatted_completion'])

            # Loop through each row in the DataFrame and format the prompts and completions
            for index, row in df.iterrows():
                prompt = row['prompt']
                completion = row['completion']
                formatted_data = pd.concat([formatted_data, self.format_prompt_completion_df(prompt, completion)], ignore_index=True)

            # Save the formatted data as JSON
            self.data_formatter.df_to_json(formatted_data, output_file)

        except json.decoder.JSONDecodeError as e:
            logger.error("\n\n", e, "\n")
            logger.error("Error parsing JSON. Skipping context ... \n\n", context)

    def generate_system_prompt(self, num_questions: int, question_type: str, detailed_explanation: bool = True):
        """
        Generates a system prompt including instructions for the number of questions and their type.

        This method creates a system prompt to guide the generation of questions of a specific type and quantity. 
        It is useful for creating structured and targeted questions for AI model training or testing.

        :param num_questions: The number of questions to be included in the prompt.
        :type num_questions: int
        :param question_type: The type of questions, such as 'open-ended', 'yes/no', etc.
        :type question_type: str
        :param detailed_explanation: Flag indicating whether to include instructions for detailed explanations and arguments. Defaults to True.
        :type detailed_explanation: bool, optional

        :return: A string containing the generated system prompt.
        :rtype: str

        :example:

        ::

            >>> generator = PromptCompletionGenerator(openai_key="sk-xxxxx")
            >>> system_prompt = generator.generate_system_prompt(5, 'open-ended', True)
            >>> print(system_prompt)
            # Outputs a generated system prompt with guidelines for 5 open-ended questions.
        """

        # Define static questions for different question types
        static_questions = {
            # Existing types
            "open-ended": ["What is the capital of France", "How does photosynthesis work", "Where is the Eiffel Tower located", "Why do birds migrate", "When did World War II end"],
            "yes-no": ["Is the sky blue", "Can you swim", "Do cats have tails", "Will it rain tomorrow", "Did you eat breakfast"],
            "closed-ended": ["On a scale of 1 to 5, how satisfied are you with our service", "Rate your agreement from 1 to 5 with the statement", "How likely are you to recommend our product from 1 to 5", "How would you rate your experience from 1 to 5", "Rate your knowledge level from 1 to 5"],
            "ranking": ["Please rank the following movies in order of preference", "Rank the cities by population size from largest to smallest", "Order the items by importance from most to least", "Rank the books in the order you read them", "Rank the colors by your favorite to least favorite"],
            "hypothetical": ["What would you do if you won an award", "In a hypothetical scenario, how would you react if you met a celebrity", "Imagine a situation where you find a wallet on the street", "What would be your response if you saw a UFO", "If you could time travel, where and when would you go"],
            "clarification": ["Could you please explain the concept of blockchain", "I need clarification on the fourth step of the process", "Can you provide more details about the theory of relativity", "Please explain the main idea of the book", "What is the meaning behind the artwork"],
            "leading": ["Don't you agree that exercise is important for a healthy lifestyle", "Isn't it true that honesty is the best policy", "Wouldn't you say that education is valuable", "Aren't you excited about the upcoming event", "Don't you think chocolate is delicious"],
            "critical-thinking": ["How would you solve the problem of climate change", "What are your thoughts on the impact of technology on society", "Can you critically analyze the economic implications of the policy", "What strategies would you use to improve customer satisfaction", "How do you propose to address the issue of poverty"],
            "reflective": ["How do you feel about your recent achievements", "Share your reflections on the past year", "What are your sentiments regarding the current political situation", "Reflect on your experiences during the trip", "How do you perceive the concept of success"],

            # Additional types
            "multiple-choice": ["Which of these is a fruit: Tomato, Potato, Broccoli", "Which planet is known as the Red Planet: Mars, Venus, Jupiter", "Who wrote 'Romeo and Juliet': Shakespeare, Dickens, Tolstoy", "Which gas is most abundant in Earth's atmosphere: Oxygen, Nitrogen, Carbon Dioxide", "What is the hardest natural substance: Diamond, Gold, Iron"],
            "scale/rating": ["Rate the difficulty of this task from 1 to 10", "On a scale from 1 to 10, how likely would you recommend our app", "Rate your level of interest in art from 1 to 10", "How would you rate the customer service on a scale of 1 to 5", "Rate your understanding of the topic before and after the lecture from 1 to 5"],
            "comparative": ["Which do you find more challenging: Math or English", "Compare your experience between online and offline shopping", "Which do you prefer: Working from home or office", "Compare the taste of apples and oranges", "Which is more beneficial: Regular exercise or a balanced diet"],
            "cause and effect": ["What caused the extinction of dinosaurs", "What effects do you think social media has on teenagers", "What led to the rise of e-commerce", "How does pollution affect marine life", "What caused the global shift towards renewable energy sources"],
            "problem-solving": ["How would you address internet privacy concerns", "What strategy would you use to improve literacy rates", "Suggest a solution for managing urban waste", "How would you enhance public transportation in cities", "Propose a method to reduce food waste in restaurants"],
            "behavioral": ["Describe a time when you had to work under pressure", "Tell me about a challenge you faced at work and how you overcame it", "Can you discuss a time when you had to resolve a conflict", "Describe an instance where you had to learn something new quickly", "Tell us about a time you led a team to achieve a goal"],
            "opinion/attitude": ["What is your stance on cloning technology", "What are your views on homeschooling", "What do you think about the impact of artificial intelligence on jobs", "What is your opinion on space exploration", "How do you feel about the current trends in fashion"],
            "experience-based": ["Describe your most memorable travel experience", "Can you share an experience where you had to make a tough decision", "Talk about a significant learning experience in your life", "Share your experience with a recent technological gadget", "Discuss an event that significantly changed your perspective"],
            "situational": ["If you were the CEO of a company, how would you increase profits", "In a situation where resources are limited, how would you prioritize tasks", "If you were stranded on an island, how would you survive", "If you were given a chance to change a decision you made, what would it be", "In a scenario where you have to work with a difficult colleague, how would you handle it"],
            "demographic": ["What is your age group: Under 18, 18-24, 25-34, 35-44, 45+", "Specify your highest level of education: High School, Bachelor's, Master's, PhD", "Indicate your employment status: Employed, Unemployed, Student, Retired", "Select your field of work: Technology, Education, Healthcare, Business, Arts", "What is your marital status: Single, Married, Divorced, Widowed"],
            "exploratory": ["What factors influence your buying decisions", "Explore the reasons behind the popularity of streaming services", "What motivates you to stay healthy and fit", "Explore the factors contributing to job satisfaction", "What drives the innovation in the smartphone industry"],
            "diagnostic": ["Identify the reasons for the decline in sales last quarter", "Diagnose the root cause of the team's communication issues", "What factors are causing the delay in project delivery", "Diagnose the reasons behind the high employee turnover rate", "Identify the challenges in implementing the new software system"],
            "sequential/process": ["Outline the steps involved in baking a cake", "Describe the process of photosynthesis in plants", "Explain the steps to resolve a customer complaint", "Detail the process of creating a mobile application", "Describe the sequence of events in a typical workday"]
        }


        # Check if the question_type is valid
        if question_type not in static_questions:
            raise ValueError("Invalid question_type. Supported values are: {}".format(", ".join(static_questions.keys())))

        # Initialize the initial prompt
        system_prompt = "Given the context below, generate a JSON array with {} precisely crafted pairs of prompts as {} questions and their corresponding completions as JSON Array, following these guidelines for the context below:\n".format(num_questions, question_type)

        # Generate example of questions based on the specified type to be added an a few-shot learning example
        for i in range(1, num_questions + 1):
            questions = static_questions[question_type]
            if i <= len(questions):
                question = questions[i - 1] + "?"
            else:
                question = "Example {}: {} question {}".format(i, question_type.capitalize(), i)
            
            #system_prompt += "Example {}: {}\n".format(i, question)

        # Add the remaining context
        #system_prompt += "Each prompt is inherently correctly answerable with an in-depth and justified response.\n"
        # Include detailed instructions based on the flag
        if detailed_explanation:
            system_prompt += "Each response to a prompt should be meticulously crafted to offer a detailed explanation along with a robust argument to substantiate the response.\n"
            system_prompt += "Each completion must be developed offering a sufficient explanation and ample arguments to justify the given response.\n"

        #system_prompt += "The returned response of all prompts should be formatted within a JSON ARRAY structure.\n"
        #system_prompt += "Each individual JSON record should encapsulate one distinct prompt and its corresponding in-depth completion.\n"
        json_prefix = """```json"""
        #system_prompt += "[LIFE CRITICAL REQUIREMENT] Output Format: prompt and completion as a JSON ARRAY structure:\n"
        system_prompt += 'EXACT JSON ARRAY  STRUCTURE FORMAT:\n[{\"prompt\": \"question1\", \"completion\": \"answer1\"}, {\"prompt\": \"question1\", \"completion\": \"answer1\"}]'
        #system_prompt += "\ndo NOT add "+json_prefix+" as prefix. \n\n"
        #system_prompt += "```json\n"
        #system_prompt += "[\n"
        #system_prompt += '    {"prompt": "question1", "completion": "answer1"},\n'
        #system_prompt += '    {"prompt": "question2", "completion": "answer2"}\n'
        #system_prompt += "]\n"
        #system_prompt += "```\n"
        

        return system_prompt


    def generate_prompt_completions(self, context_text: str, output_csv: str,
                                    temperature: float = 0.1, 
                                    model: str = "gpt-3.5-turbo-1106",
                                    max_tokens: int = 1054, 
                                    top_p: float = 1.0,
                                    frequency_penalty: float = 0.0, 
                                    presence_penalty: float = 0.0,
                                    retry_limit: int = 3,
                                    num_questions: int = 3, 
                                    question_type: str = "open-ended",
                                    detailed_explanation: bool = True) -> List[Dict[str, str]]:
        """
        Generates prompt completions using the OpenAI API and records them to a CSV file.

        :method generate_prompt_completions: Use the OpenAI model to generate responses based on provided context and record the prompt-completion pairs in a CSV file.
        :type generate_prompt_completions: method

        :param context_text: The context based on which prompts are generated.
        :type context_text: str

        :param output_csv: The file path for saving generated completions in CSV format.
        :type output_csv: str

        :param temperature: The level of randomness in the output. Higher values lead to more varied outputs. Defaults to 0.1.
        :type temperature: float, optional

        :param model: The OpenAI model used for generation. Defaults to "gpt-3.5-turbo-1106".
        :type model: str, optional

        :param max_tokens: The maximum length of the generated output. Defaults to 1054.
        :type max_tokens: int, optional

        :param top_p: The proportion of most likely tokens considered for sampling. Defaults to 1.0.
        :type top_p: float, optional

        :param frequency_penalty: The decrease in likelihood for frequently used tokens. Defaults to 0.0.
        :type frequency_penalty: float, optional

        :param presence_penalty: The decrease in likelihood for already used tokens. Defaults to 0.0.
        :type presence_penalty: float, optional

        :param retry_limit: The maximum number of retries for API call failures. Defaults to 3.
        :type retry_limit: int, optional

        :param num_questions: The number of questions to generate. Defaults to 3.
        :type num_questions: int, optional

        :param question_type: The type of questions to generate, such as "open-ended", "yes/no", etc. Defaults to "open-ended".
        :type question_type: str, optional

        :param detailed_explanation: Flag indicating whether to include instructions for detailed explanations and arguments. Defaults to True.
        :type detailed_explanation: bool, optional

        :return: A list of dictionaries, each containing 'prompt' and 'completion' keys.
        :rtype: List[Dict[str, str]]

        :raises ValueError: If the OpenAI API key is invalid or not provided.
        :raises Exception: If the function fails after the specified number of retries.

        :example:

        ::

            >>> generate_prompt_completions("Discuss the impact of AI in healthcare.", "ai_healthcare.csv")
            # Generates prompt completions based on the context about AI in healthcare and records them in 'ai_healthcare.csv'.

        :notes:
        - Proper API key authorization is essential for successful API requests. Ensure the OpenAI key is valid and has the necessary permissions.
        """

    
        # Customize the initial prompt based on the number and type of questions
        OUTPUT_FORMAT='[{"prompt": "question1", "completion": "answer1"}, {"prompt": "question1", "completion": "answer1"}]'
        system_prompt = self.generate_system_prompt(num_questions, question_type, detailed_explanation)
        #user_prompt = 'requirement:  The output must be a JSON ARRAY structure exactly like: '+ OUTPUT_FORMAT +"\ncontext: \n" + context_text
        user_prompt = "context: \n" + context_text
        # Log the function call
        redacted_key = openai.api_key[:10]+"..." 
        logger.debug(
            f"Called generate_prompt_completions with params: \n"
            f"context_text={context_text}, \noutput_csv={output_csv}, "
            f"system_prompt={system_prompt}, \nuser_prompt={user_prompt}, \n"
            f"openai_key={redacted_key}, \ntemperature={temperature}, \n"
            f"model={model}, \nmax_tokens={max_tokens}, \ntop_p={top_p}, \n"
            f"frequency_penalty={frequency_penalty}, \npresence_penalty={presence_penalty}, \n"
            f"retry_limit={retry_limit}, \nnum_questions={num_questions}, \n"
            f"question_type={question_type}"
        )


        retry_count = 0

        while retry_count < retry_limit:
            try:
                logger.debug(f"Attempting to generate prompt completions with params: \n")

                response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": user_prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=["END"]
                )
                
                logger.debug(f"[generate_prompt_completions] Successfully generated prompt completions with params: \n")
                # Process the response
                #if 'choices' in response and len(response.choices) > 0:
                json_string = response.choices[0].message.content #.strip()
                logger.debug(f"[generate_prompt_completions] Successfully processed response: \n{json_string} ...")
                # Parse the JSON string

                logger.info(f"[generate_prompt_completions] Attempting to parse JSON response: \n\n{json_string} ...")
                prompt_completion_pairs = self.data_formatter.extract_json_array(json_string) # remove ```json and ``` from the string if exists and extract the array as list of dict
                logger.info(f"[generate_prompt_completions] Successfully parsed JSON response")
                
                # Save to CSV
                list_to_save = [{"prompt": pair["prompt"], "completion": pair["completion"], "question_type": question_type} for pair in prompt_completion_pairs]
                self.data_formatter.list_to_csv(list_to_save, output_csv)
                return self.data_formatter.remove_json_markers(json_string) # remove ```json and ``` from the string and return the json string
                #else:
                #    retry_count += 1
                #    logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an error:\n")
                #    logger.error(f"[generate_prompt_completions] retry_count:Error processing response: \n{response}")
                #    time.sleep(RETRY_SLEEP_SECONDS)
                #    raise Exception(f"[generate_prompt_completions] Error processing OpenAI response: \n{response}")
            except json.JSONDecodeError as e:
                retry_count += 1
                logger.error(f"[generate_prompt_completions] Error decoding JSON response: {e}. \njson_string: {json_string}")
                logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an error:\n{e}")
                time.sleep(RETRY_SLEEP_SECONDS)

            except openai.APIConnectionError as e:
                retry_count += 1            
                logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an APIConnectionError error:\n{e}")
                logger.error(f"[generate_prompt_completions] Retrying in 0.5 seconds...")
                time.sleep(10)

            except openai.RateLimitError as e:
                # If the request fails due to rate error limit, increment the retry counter, sleep for 0.5 seconds, and then try again
                retry_count += 1
                logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} failed due to RateLimit Error {e}. Max tokens = {max_tokens}. Trying again in 0.5 seconds...")
                max_tokens = int(max_tokens * 0.8)
                time.sleep(10)  # Pause for 0.5 seconds

            except openai.APIStatusError as e:
                retry_count += 1
                logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an APIStatusError:\n{e}")
                # If the request fails due to service unavailability, sleep for 10 seconds and then try again without incrementing the retry counter
                logger.error(f"\n\n[generate_prompt_completions] Service is currently unavailable. Waiting for 10 seconds before retrying...\n\n")
                time.sleep(10)  # Pause for 10 seconds

            except AttributeError as e:
                retry_count += 1
                logger.error(f"[generate_prompt_completions] Retry attempt {retry_count} due to an AttributeError:\n{e}")
                # Handle the exception and log the error message
                logging.error("[generate_prompt_completions] An AttributeError occurred: %s", e)
                # You can also add additional error handling code here if needed

            except Exception as e:
                logger.error(f"\n\n[generate_prompt_completions] Exception: {e}\n\n")
                retry_count += 1
                time.sleep(20)
                if retry_count > 3:
                    logger.error(f"[generate_prompt_completions] Gave up after {retry_limit} attempts for context: {context_text}. Exit.")
                    raise Exception(f"[generate_prompt_completions] Gave up after {retry_limit} attempts for context: {context_text[:150]}...\n Exit.")

        if retry_count >=3:
            logger.error(f"[generate_prompt_completions] Gave up after {retry_limit} attempts for context: {context_text[:150]}...\n Exit.")
            raise Exception("[generate_prompt_completions] Gave up after max retry limit attempts ...\n Exit.")

        return self.data_formatter.remove_json_markers(json_string) # remove ```json and ``` from the string and return the json string
 

    def create_dataset(self, context_filename: str, output_filename: str,
                   temperature: float = 0.1, 
                   model: str = "gpt-3.5-turbo-1106",
                   max_tokens: int = 1054, 
                   top_p: float = 1.0,
                   frequency_penalty: float = 0.0, 
                   presence_penalty: float = 0.0,
                   retry_limit: int = 3,
                   num_questions: int = 3, 
                   question_types: List[str] = None,
                   detailed_explanation: bool = True):
        """
        Generates a dataset with various types of questions based on the provided context.

        :method create_dataset: Create a dataset by generating questions of specified types for each context in the provided CSV file.
        :type create_dataset: method

        :param context_filename: Path to the CSV file containing context data.
        :type context_filename: str

        :param output_filename: Path to save the generated dataset.
        :type output_filename: str

        :param temperature: Controls randomness in response generation. Defaults to 0.1.
        :type temperature: float, optional

        :param model: The OpenAI model to be used for generating responses. Defaults to "gpt-3.5-turbo-1106".
        :type model: str, optional

        :param max_tokens: Maximum length of the generated output. Defaults to 1054.
        :type max_tokens: int, optional

        :param top_p: Nucleus sampling parameter, controlling the range of tokens considered for generation. Defaults to 1.0.
        :type top_p: float, optional

        :param frequency_penalty: Decrease in likelihood for previously used tokens. Defaults to 0.0.
        :type frequency_penalty: float, optional

        :param presence_penalty: Decrease in likelihood for currently present tokens. Defaults to 0.0.
        :type presence_penalty: float, optional

        :param retry_limit: Maximum number of retries for API calls in case of failure. Defaults to 3.
        :type retry_limit: int, optional

        :param num_questions: Number of questions to generate for each context. Defaults to 3.
        :type num_questions: int, optional

        :param question_types: Types of questions to generate (e.g., "open-ended", "yes-no"). If not specified, defaults to various types.
        :type question_types: List[str], optional

        :param detailed_explanation: Flag to indicate whether to include detailed explanations in the generated content. Defaults to True.
        :type detailed_explanation: bool, optional

        :raises Exception: If an error occurs during question generation or file operations.

        :example:

        ::
            >>> generator = PromptCompletionGenerator(openai_key="your-api-key")
            >>> generator.create_dataset(
                context_filename="path/to/context.csv",
                output_filename="path/to/generated_dataset.csv",
                temperature=0.7,
                model="gpt-3.5-turbo",
                num_questions=5,
                question_types=["open-ended", "yes-no"],
                detailed_explanation=False
            )
            # This example generates a dataset with specified question types based on the contexts from 'path/to/context.csv'.

        :notes:
        - The method iterates over each row in the context CSV file, generating questions of specified types for each context. The generated questions and answers are saved to the output CSV file.
        """

        # Implementation of the method...

        
        if question_types is None:
            question_types = ["open-ended", "yes-no", "reflective", "closed-ended"]

        context_df = self.context_extractor.from_csv_as_df(context_filename, encoding="utf-8")

        for index, row in context_df.iterrows():
            context = row['context']
            for question_type in question_types:
              try:
                  print(f"Generating {question_type} questions for context at index {index}")
                  questions = self.generate_prompt_completions( 
                              context, output_filename,
                              temperature=temperature,
                              max_tokens=max_tokens,
                              top_p=top_p,
                              frequency_penalty=frequency_penalty,
                              presence_penalty=presence_penalty,
                              retry_limit=retry_limit,
                              num_questions=num_questions,
                              question_type=question_type,
                              model=model,
                              detailed_explanation=detailed_explanation
                          )
                  print(f'Results for {question_type}:', questions)
              except Exception as e:
                  print(f"Error: {e}")