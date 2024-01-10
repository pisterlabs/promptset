"""This module contains the Japanese to English translator functions"""

import openai
import html
from yattag import Doc
import xml.etree.ElementTree as ET

from gptwntranslator.api.openai_api import call_api, get_line_token_count, validate_model
from gptwntranslator.helpers.config_helper import Config
from gptwntranslator.helpers.data_helper import get_targeted_sub_chapters
from gptwntranslator.helpers.design_patterns_helper import singleton
from gptwntranslator.helpers.logger_helper import CustomLogger
from gptwntranslator.helpers.task_helper import Task
from gptwntranslator.models.novel import Novel
from gptwntranslator.models.chunk import Chunk
from gptwntranslator.models.sub_chapter import SubChapter
from gptwntranslator.models.term_sheet import TermSheet


logger = CustomLogger(__name__)

class GPTTranslatorException(Exception):
    pass

class GPTTranslatorAPIException(GPTTranslatorException):
    pass

class GPTTranslatorAPIRetryableException(GPTTranslatorAPIException):
    pass

class GPTTranslatorAPINoRetriesException(GPTTranslatorAPIException):
    pass

class GPTTranslatorGPTFormatException(GPTTranslatorException):
    pass

class GPTTranslator:    
    def __init__(self) -> None:
        TypeError(f"'{self.__class__.__name__}' cannot be instantiated. Create a subclass instead.")
    
    def _initialize(self, available_models: dict, terms_models: list[str], translation_models: list[str], summary_models: list[str], metadata_models: list[str], original_language: str="Japanese", target_language: str="English") -> None:
        
        # Validate the parameters
        if not isinstance(available_models, dict):
            raise TypeError("Available models must be a dictionary")
        if not all(validate_model(model) for model in available_models.values()):
            raise ValueError("Available models must be a dictionary of valid models")
        if not isinstance(terms_models, list):
            raise TypeError("Terms models must be a list")
        if not all(isinstance(terms_model, str) for terms_model in terms_models):
            raise TypeError("Terms models must be a list of strings")
        if not all(terms_model in available_models for terms_model in terms_models):
            raise ValueError("Terms models must be a list of valid models")
        if not isinstance(translation_models, list):
            raise TypeError("Translation models must be a list")
        if not all(isinstance(translation_model, str) for translation_model in translation_models):
            raise TypeError("Translation models must be a list of strings")
        if not all(translation_model in available_models for translation_model in translation_models):
            raise ValueError("Translation models must be a list of valid models")
        if not isinstance(summary_models, list):
            raise TypeError("Summary models must be a list")
        if not all(isinstance(summary_model, str) for summary_model in summary_models):
            raise TypeError("Summary models must be a list of strings")
        if not all(summary_model in available_models for summary_model in summary_models):
            raise ValueError("Summary models must be a list of valid models")
        if not isinstance(metadata_models, list):
            raise TypeError("Metadata models must be a list")
        if not all(isinstance(metadata_model, str) for metadata_model in metadata_models):
            raise TypeError("Metadata models must be a list of strings")
        if not all(metadata_model in available_models for metadata_model in metadata_models):
            raise ValueError("Metadata models must be a list of valid models")
        if not isinstance(original_language, str):
            raise TypeError("Original language must be a string")
        if not isinstance(target_language, str):
            raise TypeError("Target language must be a string")
        
        self._available_models = available_models
        self._terms_models = terms_models
        self._translation_models = translation_models
        self._summary_models = summary_models
        self._metadata_models = metadata_models
        self._original_language = original_language
        self._target_language = target_language
        
        cf = Config()
        self._original_language_str = cf.get_language_name_for_code(original_language) if original_language in cf.get_languages() else ""
        self._target_language_str = cf.get_language_name_for_code(target_language) if target_language in cf.get_languages() else ""

    def set_original_language(self, original_language: str) -> None:
        logger.debug(f"Setting original language to '{original_language}'")
        # Validate the parameters
        if not isinstance(original_language, str):
            logger.error(f"Original language ({original_language}) must be a string")
            raise TypeError(f"Original language ({original_language}) must be a string")
        
        cf = Config()

        # Verify language exists in config
        if original_language not in cf.get_languages():
            logger.error(f"Original language ({original_language}) must be a valid language in ({cf.get_languages()})")
            raise ValueError(f"Original language ({original_language}) must be a valid language in ({cf.get_languages()})")

        self._original_language = original_language
        self._original_language_str = cf.get_language_name_for_code(original_language) if original_language in cf.get_languages() else ""

    def _get_api_model(self, model: str) -> dict:
        # Validate the parameters
        if not isinstance(model, str):
            logger.error(f"Model ({model}) must be a string")
            raise TypeError("Model must be a string")
        if model not in self._available_models:
            logger.error(f"Model ({model}) must be a valid model")
            raise ValueError("Model must be a valid model")
        
        # Get the translation model
        return self._available_models[model]
    
    def _split_text_into_chunks(self, text: str, division_size: int, line_token_counts: list[int]) -> list[str]:
        # Validate the parameters
        if not isinstance(text, str):
            logger.error(f"Text must be a string")
            raise TypeError("Text must be a string")
        if not isinstance(division_size, int):
            logger.error(f"Division size ({division_size}) must be an integer")
            raise TypeError("Division size must be an integer")
        if division_size <= 0:
            logger.error(f"Division size ({division_size}) must be greater than 0")
            raise ValueError("Division size must be greater than 0")
        if not isinstance(line_token_counts, list):
            logger.error(f"Line token counts ({line_token_counts}) must be a list")
            raise TypeError("Line token counts must be a list")
        if not all(isinstance(line_token_count, int) for line_token_count in line_token_counts):
            logger.error(f"Line token counts ({line_token_counts}) must be a list of integers")
            raise TypeError("Line token counts must be a list of integers")
        if not all(line_token_count >= 0 for line_token_count in line_token_counts):
            logger.error(f"Line token counts ({line_token_counts}) must be a list of positive integers")
            raise ValueError("Line token counts must be a list of positive integers")
        
        # Initialize some variables
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Split the text into chunks
        for line, token_count in zip(text.splitlines(), line_token_counts):
            if current_tokens + token_count <= division_size:
                current_chunk.append(line)
                current_tokens += token_count
            else:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_tokens = token_count

        # Add the last chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks
    
    def _estimate_chunks(self, total_lines: int, division_size: int, line_token_counts: list[int]) -> int:
        # Validate the input
        if not isinstance(total_lines, int):
            logger.error(f"Total lines ({total_lines}) must be an integer")
            raise TypeError("total_lines must be an integer.")
        if total_lines <= 0:
            logger.error(f"Total lines ({total_lines}) must be a positive integer")
            raise ValueError("total_lines must be a positive integer.")
        if not isinstance(division_size, int):
            logger.error(f"Division size ({division_size}) must be an integer")
            raise TypeError("division_size must be an integer.")
        if division_size <= 0:
            logger.error(f"Division size ({division_size}) must be a positive integer")
            raise ValueError("division_size must be a positive integer.")
        if not isinstance(line_token_counts, list):
            logger.error(f"Line token counts ({line_token_counts}) must be a list")
            raise TypeError("line_token_counts must be a list.")
        if not all(isinstance(token_count, int) for token_count in line_token_counts):
            logger.error(f"Line token counts ({line_token_counts}) must be a list of integers")
            raise TypeError("line_token_counts must be a list of integers.")
        if not all(token_count >= 0 for token_count in line_token_counts):
            logger.error(f"Line token counts ({line_token_counts}) must be a list of positive integers")
            raise ValueError("line_token_counts must be a list of positive integers.")
        if len(line_token_counts) != total_lines:
            logger.error(f"Line token counts ({line_token_counts}) must have the same length as total lines ({total_lines})")
            raise ValueError("line_token_counts must have the same length as total_lines.")
        
        # Estimate the number of chunks
        chunks = self._split_text_into_chunks('\n'.join([''] * total_lines), division_size, line_token_counts)

        return len(chunks)
    
    def _original_language_token_limit_worst_case(self, N: int, worst_case_ratio: float|int=1.125, safety_factor: float|int=0.8) -> int:
        # Validate the input
        if not isinstance(N, int):
            logger.error(f"N ({N}) must be an integer")
            raise TypeError("N must be an integer.")
        if N <= 0:
            logger.error(f"N ({N}) must be a positive integer")
            raise ValueError("N must be a positive integer.")
        if not isinstance(worst_case_ratio, float) and not isinstance(worst_case_ratio, int):
            logger.error(f"Worst case ratio ({worst_case_ratio}) must be a float or an integer")
            raise TypeError("worst_case_ratio must be a float or an integer.")
        if worst_case_ratio <= 0:
            logger.error(f"Worst case ratio ({worst_case_ratio}) must be a positive float or integer")
            raise ValueError("worst_case_ratio must be a positive float or integer.")
        if not isinstance(safety_factor, float) and not isinstance(safety_factor, int):
            logger.error(f"Safety factor ({safety_factor}) must be a float or an integer")
            raise TypeError("safety_factor must be a float or an integer.")
        if safety_factor <= 0:
            logger.error(f"Safety factor ({safety_factor}) must be a positive float or integer")
            raise ValueError("safety_factor must be a positive float or integer.")
        
        # Calculate the token limit
        token_limit = N * safety_factor / (1 + worst_case_ratio)
        token_limit = int(token_limit)

        return token_limit
    
    def _greedy_find_max_optimal_configuration(self, line_token_counts: list[int]) -> tuple[int, int, int, str, str, str]:
        logger.debug(f"Finding max optimal configuration for '{line_token_counts}' line token counts")

        # Validate the input
        if not isinstance(line_token_counts, list):
            logger.error(f"Line token counts ({line_token_counts}) must be a list")
            raise TypeError("line_token_counts must be a list.")
        if not all(isinstance(token_count, int) for token_count in line_token_counts):
            logger.error(f"Line token counts ({line_token_counts}) must be a list of integers")
            raise TypeError("line_token_counts must be a list of integers.")
        if not all(token_count >= 0 for token_count in line_token_counts):
            logger.error(f"Line token counts ({line_token_counts}) must be a list of positive integers")
            raise ValueError("line_token_counts must be a list of positive integers.")

        # Initialize some values
        total_lines = len(line_token_counts)
        min_cost = float('inf')
        best_combination = None
        wrapper_percentage = 0.3

        # Get the models to use
        terms_models = [self._get_api_model(model) for model in self._terms_models]
        translation_models = [self._get_api_model(model) for model in self._translation_models]
        summary_models = [self._get_api_model(model) for model in self._summary_models]

        # Iterate through all possible combinations of models and token limits
        for term_model in terms_models:
            term_model_limit = self._original_language_token_limit_worst_case(term_model["max_tokens"])
            term_model_limit = term_model_limit - (term_model_limit % 4)

            for translation_model in translation_models:
                translation_model_limit = self._original_language_token_limit_worst_case(translation_model["max_tokens"])
                translation_model_limit = translation_model_limit - (translation_model_limit % 4)

                for summary_model in summary_models:
                    summary_model_limit = self._original_language_token_limit_worst_case(summary_model["max_tokens"])
                    summary_model_limit = summary_model_limit - (summary_model_limit % 4)

                    for term_division in range(term_model_limit // 2, term_model_limit + 1, 100):
                        for translation_division in range(translation_model_limit // 2, translation_model_limit + 1, 100):
                            for summary_division in range(summary_model_limit // 2, summary_model_limit + 1, 100):
                                # Calculate the cost of the current configuration
                                term_chunks = self._estimate_chunks(total_lines, term_division, line_token_counts)
                                translation_chunks = self._estimate_chunks(total_lines, translation_division, line_token_counts)
                                summary_chunks = self._estimate_chunks(total_lines, summary_division, line_token_counts)

                                # term_cost = term_chunks * term_division * term_model["cost_per_1k_tokens"]
                                # translation_cost = translation_chunks * translation_division * translation_model["cost_per_1k_tokens"]
                                # summary_cost = summary_chunks * summary_division * summary_model["cost_per_1k_tokens"]

                                term_cost = term_chunks * (term_division * (1 + wrapper_percentage)) * term_model["cost_per_1k_tokens"]
                                translation_cost = translation_chunks * (translation_division * (1 + wrapper_percentage)) * translation_model["cost_per_1k_tokens"]
                                summary_cost = summary_chunks * (summary_division * (1 + wrapper_percentage)) * summary_model["cost_per_1k_tokens"]

                                total_cost = term_cost + translation_cost + summary_cost

                                # Check if the current configuration is better than the best configuration so far, and update the best configuration if it is
                                if total_cost < min_cost:
                                    min_cost = total_cost
                                    best_combination = (term_division, translation_division, summary_division, term_model['name'], translation_model['name'], summary_model['name'])

        logger.info(f"Found max optimal configuration: {best_combination}")
        return best_combination
    
    def _calculate_line_token_counts(self, text: str) -> list[int]:
        # Validate input
        if not isinstance(text, str):
            logger.error(f"Text ({text}) must be a string")
            raise TypeError("Text must be a string")
        
        # Split the text into lines and calculate the token count for each line
        lines = text.splitlines()
        line_token_counts = [get_line_token_count(line) for line in lines]

        return line_token_counts
    
    def _handle_api_exceptions(self, e: Exception) -> None:
        if isinstance(e, openai.error.APIError):
            raise GPTTranslatorAPIRetryableException(e)
        elif isinstance(e, openai.error.Timeout):
            raise GPTTranslatorAPIRetryableException(e)
        elif isinstance(e, openai.error.RateLimitError):
            raise GPTTranslatorAPIRetryableException(e)
        elif isinstance(e, openai.error.APIConnectionError):
            raise GPTTranslatorAPIRetryableException(e)
        elif isinstance(e, openai.error.InvalidRequestError):
            raise GPTTranslatorAPINoRetriesException(e)
        elif isinstance(e, openai.error.AuthenticationError):
            raise GPTTranslatorAPINoRetriesException(e)
        elif isinstance(e, openai.error.ServiceUnavailableError):
            raise GPTTranslatorAPIRetryableException(e)
        else:
            raise e
    
    def _perform_relevant_terms_action(self, **kwargs) -> str:
        logger.debug(f"Performing relevant terms action.")

        available_models = [self._get_api_model(model)['name'] for model in self._terms_models]

        chunk = kwargs["chunk"]
        model = kwargs["model"]

        # Validate input
        if not isinstance(chunk, Chunk):
            logger.error(f"Chunk ({chunk}) must be a Chunk object")
            raise TypeError("Chunk must be a Chunk object")
        if not isinstance(model, str):
            logger.error(f"Model ({model}) must be a string")
            raise TypeError("Model must be a string")
        if model not in available_models:
            logger.error(f"Model ({model}) must be a valid model. Available models: {', '.join(available_models)}")
            raise ValueError(f"Model must be a valid model. Available models: {', '.join(available_models)}")

        string_term_original_language = self._original_language_str.lower() + "_term"
        string_term_target_language = self._target_language_str.lower() + "_term"
        phonetic_term = "phonetic_for_" + string_term_original_language if self._original_language_str != "Japanese" else "rōmaji_for_" + string_term_original_language

        system_content = f'''
        You are a translator that translates novels from one language to another.

        You will be provided with:
        1) Original language
        2) Destination language
        3) A chunk of text in the original language

        You will be asked to:
        1) Generate a term list for the text you are provided
        2) Focus on proper nouns and technical terms
        3) Follow the format "- {string_term_original_language} ({phonetic_term}) - {string_term_target_language}"
        4) Maintain the novel translation format conventions

        Example:
        [Original language]
        English
        [Destination language]
        Spanish
        [Text]
        This is Darth Vader, right hand of the Emperor. His face is obscured by his flowing black robes and grotesque breath mask, which stands out next to the fascist white armored suits of the Imperial stormtroopers. Everyone instinctively backs away from the imposing warrior and a deathly quiet sweeps through the Rebel troops.
        [End]

        Expected output:
        - Darth Vader (darth vader) - Darth Vader
        - Emperor (emperor) - Emperador
        - stormtroopers (stormtroopers) - soldados imperiales
        - Rebel troops (rebel troops) - tropas rebeldes
        '''

        user_content = f'''
        [Original language]
        {self._original_language_str}
        [Destination language]
        {self._target_language_str}
        [Text]
        {chunk.contents}
        [End]
        '''

        # Build the messages list for the API call
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

        # Call the API
        try:
            logger.debug(f"Calling API.")
            response = call_api(messages, model)
            response = response['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error performing relevant terms action: {e}")
            self._handle_api_exceptions(e)
        
        logger.debug(f"Response: {response}")

        return response
    
    def _perform_translation_action(self, **kwargs) -> str:
        logger.debug(f"Performing translation action.")

        available_models = [self._get_api_model(model)['name'] for model in self._translation_models]

        chunk = kwargs["chunk"]
        term_lists = kwargs["term_lists"]
        summary = kwargs["summary"]
        translation_model = kwargs["translation_model"]
        
        # Validate inputs
        if not isinstance(chunk, Chunk):
            logger.error(f"Chunk ({chunk}) must be a Chunk object")
            raise TypeError("Chunk must be a Chunk object")
        if not isinstance(term_lists, TermSheet):
            logger.error(f"Term lists must be a TermSheet object")
            raise TypeError("Term lists must be a TermSheet object")
        if not isinstance(summary, str):
            logger.error(f"Summary must be a string")
            raise TypeError("Summary must be a string")
        if not isinstance(translation_model, str):
            logger.error(f"Translation model ({translation_model}) must be a string")
            raise TypeError("Translation model must be a string")
        if translation_model not in available_models:
            logger.error(f"Translation model ({translation_model}) must be a valid model. Available models: {', '.join(available_models)}")
            raise ValueError(f"Translation model must be a valid model. Available models: {', '.join(available_models)}")
        

        system_message = f'''
        You are a translator that translates novels from one language to another.

        You will be provided with:
        1) Original language
        2) Destination language
        3) The preceding line before the text
        4) The following line after the text
        5) A list of relevant terms and their translations
        6) A summary of the enclosing section
        7) A chunk of text in the original language

        You will be asked to:
        1) Translate the text
        2) Maintain the novel translation format conventions
        3) Respect the context provided
        4) Use the relevant terms list to translate technical terms and proper nouns

        Example:
        [Original language]
        English
        [Destination language]
        Spanish
        [Preceding line]
        This is Darth Vader, right hand of the Emperor.
        [Following line]
        Everyone instinctively backs away from the imposing warrior and a deathly quiet sweeps through the Rebel troops.
        [Relevant terms]
        - Darth Vader (darth vader) - Darth Pato
        - stormtroopers (stormtroopers) - muñecos de nieve
        - Rebel troops (rebel troops) - tropas rebeldes
        [Summary]
        A Description of Darth Vader's appearance.
        [Text]
        His face is obscured by his flowing black robes and grotesque breath mask, which stands out next to the fascist white armored suits of the Imperial stormtroopers.
        [End]

        Expected output:
        Su cara está oculta por sus largas túnicas negras y grotesco respirador, que destaca junto a los trajes blancos de armadura fascistas de los muñecos de nieve imperiales.
        '''

        user_message = f'''
        [Original language]
        {self._original_language_str}
        [Destination language]
        {self._target_language_str}
        [Preceding line]
        {chunk.prev_line if chunk.prev_line else ""}
        [Following line]
        {chunk.next_line}
        [Relevant terms]
        {term_lists.for_api(chunk.contents, self._original_language)}
        [Summary]
        {summary}
        [Text]
        {chunk.contents}
        [End]
        '''

        
        # Build the messages to send to the API
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        # # Build the messages to send to the API
        # messages = [
        #     {"role": "system", "content": f"You are a {self._original_language_str} to {self._target_language_str} translator."},
        #     {"role": "user", "content": 
        #         f"Translate the chunk of text I'm about to provide. Mantain {self._original_language_str} novel translation format convetions. As context, I'll provide the immediate previous line of the text, the immediate next line of the text, the summary of the enclosing section, and a list of relevant terms and their translations to use. Do not repeat the {self._original_language_str} text, nor clarify your actions."},
        #     {"role": "assistant", "content": "Understood. Please provide the previous line."}
        # ]
        # if chunk.prev_line:
        #     messages.append({"role": "user", "content": chunk.prev_line})
        #     messages.append({"role": "assistant", "content": "Understood. Please provide the next line."})
        # else:
        #     messages.append({"role": "user", "content": ""})
        #     messages.append({"role": "assistant", "content": "No previous line provided. The text is the first line of the text. Please provide the next line."})

        # if chunk.next_line:
        #     messages.append({"role": "user", "content": chunk.next_line})
        #     messages.append({"role": "assistant", "content": "Understood. Please provide the summary."})
        # else:
        #     messages.append({"role": "user", "content": ""})
        #     messages.append({"role": "assistant", "content": "No next line provided. The text is the last line of the text. Please provide the summary."})
        # if summary:
        #     messages.append({"role": "user", "content": summary})
        #     messages.append({"role": "assistant", "content": "Understood. Please provide the relevant terms."})
        # else:
        #     messages.append({"role": "user", "content": ""})
        #     messages.append({"role": "assistant", "content": "No summary provided. The text is the first line of the text. Please provide the relevant terms."})
        # messages.append({"role": "user", "content": term_lists.for_api(chunk.contents, self._original_language)})
        # messages.append({"role": "assistant", "content": f"Understood. Please provide the text. I'll translate it to {self._target_language_str}."})
        # messages.append({"role": "user", "content": chunk.contents})

        # Call the API
        try:
            logger.debug(f"Calling API.")
            response = call_api(messages, model=translation_model)
            response = response['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error performing translation action: {e}")
            self._handle_api_exceptions(e)

        logger.debug(f"Response: {response}")
            
        return response
    
    def _perform_summary_action(self, **kwargs) -> str:
        logger.debug(f"Performing summary action.")

        available_models = [self._get_api_model(model)['name'] for model in self._summary_models]

        chunk = kwargs['chunk']
        previous_summary = kwargs['previous_summary']
        summarization_model = kwargs['summarization_model']

        # Validate parameters
        if not isinstance(chunk, str):
            logger.error(f"Chunk must be a string")
            raise TypeError("Chunk must be a string")
        if not isinstance(previous_summary, str):
            logger.error(f"Previous summary must be a string")
            raise TypeError("Previous summary must be a string")
        if not isinstance(summarization_model, str):
            logger.error(f"Summarization model must be a string")
            raise TypeError("Summarization model must be a string")
        if summarization_model not in available_models:
            logger.error(f"Summarization model ({summarization_model}) must be a valid model. Available models: {', '.join(available_models)}")
            raise ValueError(f"Summarization model must be a valid model. Available models: {', '.join(available_models)}")
        
        system_message = f'''
        You are an assistant updates the summary of an ongoing text.

        You will be provided with:
        1) The previous summary up to this point.
        2) The next section of the text to summarize.

        You will be asked to:
        1) Provide a new summary of the ongoing text.
        2) Stay within a few lines of text.
        3) Keep the summary concise.

        Example:
        [Previous summary]
        A Description of Darth Vader's appearance.
        [Next section]
        Everyone instinctively backs away from the imposing warrior and a deathly quiet sweeps through the Rebel troops.
        [End]

        Expected output:
        Darth Vader is introduced and the rebels are scared of him.
        '''

        user_message = f'''
        [Previous summary]
        {previous_summary}
        [Next section]
        {chunk}
        [End]
        '''

        # Build the messages to send to the API
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        # # Build the messages to send to the API
        # messages = [
        #     {"role": "system", "content": f"You are an assistant that summarizes {self._original_language_str} text in {self._target_language_str}."},
        #     {"role": "user", "content": f"You are summarizing a text. Part of this text has already been summarized. I'll provide this previous summary and the proceeding chunk text. Please provide an updated summary of the text. Do not repeat the {self._original_language_str} text, nor clarify your actions."},
        #     {"role": "assistant", "content": "Understood. Please provide the previous summary."}
        # ]
        # if previous_summary:
        #     messages.append({"role": "user", "content": previous_summary})
        #     messages.append({"role": "assistant", "content": f"Understood. Please provide the text. I'll summarize it in {self._target_language_str}."})
        # else:
        #     messages.append({"role": "user", "content": ""})
        #     messages.append({"role": "assistant", "content": f"No summary provided. The text is the first line of the text. Please provide the text. I'll summarize it in {self._target_language_str}."})
        # messages.append({"role": "user", "content": chunk})

        # Call the API
        try:
            logger.debug(f"Calling API.")
            response = call_api(messages, model=summarization_model)
            response = response['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error performing summary action: {e}")
            self._handle_api_exceptions(e)

        logger.debug(f"Summary action response: {response}")
        
        return response
    
    def _perform_novel_metadata_action(self, **kwargs) -> None:
        logger.debug(f"Performing novel metadata action.")

        available_models = [self._get_api_model(model)['name'] for model in self._metadata_models]

        novel = kwargs['novel']
        metadata_model = kwargs['metadata_model']

        # Validate parameters
        if not isinstance(novel, Novel):
            logger.error(f"Novel must be a Novel object")
            raise TypeError("Novel must be a Novel object")
        if not isinstance(metadata_model, str):
            logger.error(f"Metadata model must be a string")
            raise TypeError("Metadata model must be a string")
        if metadata_model not in available_models:
            logger.error(f"Metadata model ({metadata_model}) must be a valid model. Available models: {', '.join(available_models)}")
            raise ValueError(f"Metadata model must be a valid model. Available models: {', '.join(available_models)}")

        doc, tag, text = Doc().tagtext()

        with tag('metadata'):
            with tag('title'):
                text(novel.title)
            with tag('author'):
                text(novel.author)
            with tag('description'):
                text(novel.description)

        metadata = doc.getvalue()

        system_message = f'''
        You are a translator that translates novels from one language to another.

        You will be provided with:
        1) Original language
        2) Destination language
        3) The metadata of the novel to translate in XML format.

        You will be asked to:
        1) Translate the metadata to the destination language.
        2) Keep the XML format.
        3) Do not modify the XML structure.

        Example:
        [Original language]
        English
        [Destination language]
        Spanish
        [Metadata]
        <metadata>
            <title>Star Wars</title>
            <author>George Lucas</author>
            <description>A long time ago in a galaxy far, far away...</description>
        </metadata>
        [End]

        Expected output:
        <metadata>
            <title>Guerra de las Galaxias</title>
            <author>George Lucas</author>
            <description>Hace mucho tiempo en una galaxia muy, muy lejana...</description>
        </metadata>
        '''

        user_message = f'''
        [Original language]
        {self._original_language_str}
        [Destination language]
        {self._target_language_str}
        [Metadata]
        {metadata}
        [End]
        '''

        # Build the messages to send to the API
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        # # Build the messages to send to the API
        # messages = [
        #     {"role": "system", "content": f"You are a {self._original_language_str} to {self._target_language_str} translator."},
        #     {"role": "user", "content": 
        #         f"You are translating a novel's metadata. I'll provide the novel's metadata in {self._original_language_str}. Please translate the metadata to {self._target_language_str}. Do not repeat the {self._original_language_str} text, nor clarify your actions. Mantain the xml format."},
        #     {"role": "assistant", "content": f"Understood. Please provide the metadata in {self._original_language_str}. I'll translate it to {self._target_language_str}."},
        #     {"role": "user", "content": f"{metadata}"}
        # ]

        # Call the API
        try:
            logger.debug(f"Calling API.")
            response = call_api(messages, model=metadata_model)
            response = response['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error performing novel metadata action: {e}")
            self._handle_api_exceptions(e)

        logger.debug(f"Response: {response}")

        try:
            logger.debug(f"Parsing response.")
            root = ET.fromstring(response)
            novel_title_translation = novel.title_translation.copy()
            novel_author_translation = novel.author_translation.copy()
            novel_description_translation = novel.description_translation.copy()
            novel_title_translation[self._target_language] = root.find('title').text
            novel_author_translation[self._target_language] = root.find('author').text
            novel_description_translation[self._target_language] = root.find('description').text
            novel.title_translation = novel_title_translation
            novel.author_translation = novel_author_translation
            novel.description_translation = novel_description_translation
            logger.debug(f"Novel metadata updated.")
            logger.debug(f"New title: {novel.title_translation[self._target_language]}")
            logger.debug(f"New author: {novel.author_translation[self._target_language]}")
            logger.debug(f"New description: {novel.description_translation[self._target_language]}")
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            raise GPTTranslatorGPTFormatException("Invalid metadata format {}".format(e))
    
    def _perform_chapters_metadata_action(self, **kwargs) -> str:
        logger.debug(f"Performing chapters metadata action.")

        available_models = [self._get_api_model(model)['name'] for model in self._metadata_models]

        novel = kwargs['novel']
        sub_chapters = kwargs['sub_chapters']
        metadata_model = kwargs['metadata_model']

        # Validate parameters
        if not isinstance(novel, Novel):
            logger.error(f"Novel must be a Novel object")
            raise TypeError("Novel must be a Novel object")
        if not isinstance(metadata_model, str):
            logger.error(f"Metadata model must be a string")
            raise TypeError("Metadata model must be a string")
        if metadata_model not in available_models:
            logger.error(f"Metadata model ({metadata_model}) must be a valid model. Available models: {', '.join(available_models)}")
            raise ValueError(f"Metadata model must be a valid model. Available models: {', '.join(available_models)}")
        if not isinstance(sub_chapters, list):
            logger.error(f"Sub chapters must be a list")
            raise TypeError("Sub chapters must be a list")
        if not all(isinstance(sub_chapter, SubChapter) for sub_chapter in sub_chapters):
            logger.error(f"Sub chapters must be SubChapter objects")
            raise TypeError("Sub chapters must be SubChapter objects")
        # if not isinstance(targets, dict):
        #     raise TypeError("Targets must be a dictionary")
        # if not all(isinstance(key, str) for key in targets.keys()):
        #     raise TypeError("Chapter numbers must be strings")
        # if not all(key.isdigit() for key in targets.keys()):
        #     raise TypeError("Chapter numbers must be digits as strings")
        # if not all(isinstance(value, list) for value in targets.values()):
        #     raise TypeError("Sub chapter numbers must be lists")
        # if not all(isinstance(item, str) for value in targets.values() for item in value):
        #     raise TypeError("Sub chapter numbers must be strings")
        # if not all(item.isdigit() for value in targets.values() for item in value):
        #     raise TypeError("Sub chapter numbers must be digits as strings")

        # sub_chapters = get_targeted_sub_chapters(novel, targets)

        doc, tag, text = Doc().tagtext()

        with tag('metadata'):
            for chapter in novel.chapters:
                with tag('chapter', id=str(chapter.chapter_index)):
                    with tag('title'):
                        text(chapter.name)
                    for sub_chapter in chapter.sub_chapters:
                        if sub_chapter in sub_chapters:
                            with tag('sub_chapter', id=str(sub_chapter.sub_chapter_index)):
                                with tag('title'):
                                    text(sub_chapter.name)

        metadata = doc.getvalue()

        system_message = f'''
        You are a translator that translates novels from one language to another.

        You will be provided with:
        1) Original language
        2) Destination language
        3) The metadata of the novel to translate in XML format.

        You will be asked to:
        1) Translate the metadata to the destination language.
        2) Keep the XML format.
        3) Do not modify the XML structure.

        Example:
        [Original language]
        English
        [Destination language]
        Spanish
        [Metadata]
        <metadata>
            <chapter id="1">
                <title>Prologue</title>
                <sub_chapter id="1">
                    <title>The trip</title>
                </sub_chapter>
                <sub_chapter id="2">
                    <title>The thieves</title>
                </sub_chapter>
            </chapter>
            <chapter id="2">
                <title>Animals</title>
                <sub_chapter id="1">
                    <title>The hamster</title>
                </sub_chapter>
            </chapter>
        </metadata>
        [End]

        Expected output:
        <metadata>
            <chapter id="1">
                <title>Prologo</title>
                <sub_chapter id="1">
                    <title>El viaje</title>
                </sub_chapter>
                <sub_chapter id="2">
                    <title>Los ladrones</title>
                </sub_chapter>
            </chapter>
            <chapter id="2">
                <title>Animales</title>
                <sub_chapter id="1">
                    <title>El hámster</title>
                </sub_chapter>
            </chapter>
        </metadata>
        '''

        user_message = f'''
        [Original language]
        {self._original_language_str}
        [Destination language]
        {self._target_language_str}
        [Metadata]
        {metadata}
        [End]
        '''

        # Build the messages to send to the API
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        # # Build the messages to send to the API
        # messages = [
        #     {"role": "system", "content": f"You are a {self._original_language_str} to {self._target_language_str} translator."},
        #     {"role": "user", "content": 
        #         f"You are translating a novel's metadata. I'll provide the novel's metadata in {self._original_language_str}. Please translate the metadata to {self._target_language_str}. Do not repeat the {self._original_language_str} text, nor clarify your actions. Mantain the xml format."},
        #     {"role": "assistant", "content": f"Understood. Please provide the metadata in {self._original_language_str}. I'll translate it to {self._target_language_str}."},
        #     {"role": "user", "content": html.escape(metadata)}
        # ]

        # Call the API
        try:
            logger.debug(f"Calling API.")
            response = call_api(messages, model=metadata_model)
            response = html.unescape(response['choices'][0]['message']['content'])
        except Exception as e:
            logger.error(f"Error performing novel metadata action: {e}")
            self._handle_api_exceptions(e)

        logger.debug(f"Response: {response}")

        try:
            logger.debug(f"Parsing response.")
            root = ET.fromstring(response)
            chapters = root.findall('chapter')
            for chapter_data in chapters:
                chapter = novel.get_chapter(int(chapter_data.attrib['id']))
                new_translated_name = chapter.translated_name.copy()
                new_translated_name[self._target_language] = chapter_data.find('title').text
                sub_chapters = chapter_data.findall('sub_chapter')
                for sub_chapter_data in sub_chapters:
                    sub_chapter = chapter.get_sub_chapter(int(sub_chapter_data.attrib['id']))
                    new_translated_sub_name = sub_chapter.translated_name.copy()
                    new_translated_sub_name[self._target_language] = sub_chapter_data.find('title').text
                    sub_chapter.translated_name = new_translated_sub_name
                    logger.debug(f"Assigned new translated name to chapter {chapter.chapter_index} sub chapter {sub_chapter.sub_chapter_index}")
                    logger.debug(f"New translated name: {sub_chapter.translated_name[self._target_language]}")
                chapter.translated_name = new_translated_name
                logger.debug(f"Assigned new translated name to chapter {chapter.chapter_index}")
                logger.debug(f"New translated name: {chapter.translated_name[self._target_language]}")
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            raise GPTTranslatorGPTFormatException("Invalid metadata format")
        
        return None

    def _summarize_sub_chapter(self,  **kwargs) -> str:
        logger.debug(f"Summarizing sub chapter.")

        available_models = [self._get_api_model(model)['name'] for model in self._summary_models]

        chunks = kwargs['chunks']
        model = kwargs['model']

        # Validate the provided arguments
        if not isinstance(chunks, list):
            logger.error(f"Chunks must be a list")
            raise TypeError("Chunks must be a list")
        if not all(isinstance(chunk, Chunk) for chunk in chunks):
            logger.error(f"Chunks must be a list of Chunk objects")
            raise TypeError("Chunks must be a list of Chunk objects")
        if not isinstance(model, str):
            logger.error(f"Metadata model must be a string")
            raise TypeError("Metadata model must be a string")
        if model not in available_models:
            logger.error(f"Metadata model ({model}) must be a valid model. Available models: {', '.join(available_models)}")
            raise ValueError(f"Metadata model must be a valid model. Available models: {', '.join(available_models)}")

        task = Task(max_workers=4, retry_on_exceptions=(GPTTranslatorAPIRetryableException))

        previous_summary = ""
        for chunk in chunks:
            result = self._perform_summary_action(chunk=chunk.contents, summarization_model=model, previous_summary=previous_summary)
            if isinstance(result, Exception):
                raise result
            previous_summary = result
        
        logger.debug(f"Summary: {previous_summary}")

        return chunks[0].chapter_index, chunks[0].sub_chapter_index, previous_summary
    
    def _gather_terms_for_sub_chapter(self,  **kwargs) -> str:
        logger.debug(f"Gathering terms for sub chapter.")

        available_models = [self._get_api_model(model)['name'] for model in self._terms_models]

        chunks = kwargs['chunks']
        model = kwargs['model']

        # Validate the provided arguments
        if not isinstance(chunks, list):
            logger.error(f"Chunks must be a list")
            raise TypeError("Chunks must be a list")
        if not all(isinstance(chunk, Chunk) for chunk in chunks):
            logger.error(f"Chunks must be a list of Chunk objects")
            raise TypeError("Chunks must be a list of Chunk objects")
        if not isinstance(model, str):
            logger.error(f"Metadata model must be a string")
            raise TypeError("Metadata model must be a string")
        if model not in available_models:
            logger.error(f"Metadata model ({model}) must be a valid model. Available models: {', '.join(available_models)}")
            raise ValueError(f"Metadata model must be a valid model. Available models: {', '.join(available_models)}")

        task = Task(max_workers=4, retry_on_exceptions=(GPTTranslatorAPIRetryableException))

        previous_terms = {}
        for chunk in chunks:
            sub_task = task.add_subtask(self._perform_relevant_terms_action, chunk=chunk, model=model)
            previous_terms[(sub_task, chunk.chunk_index)] = ""

        sorted_keys = sorted(previous_terms.keys(), key=lambda x: x[1])
        previous_terms = {key[0]: previous_terms[key] for key in sorted_keys}

        results = task.run_subtasks()

        logger.debug("Getting terms for sub chapter complete.")
        logger.debug(f"Results: {results}")

        for task_id in previous_terms:
            result = results[task_id]
            if isinstance(result, Exception):
                raise result
            previous_terms[task_id] = result
            
        return "\n\n".join(previous_terms.values())
    
    def _translate_sub_chapter(self,  **kwargs) -> str:
        logger.debug(f"Translating sub chapter.")

        available_models = [self._get_api_model(model)['name'] for model in self._translation_models]

        chunks = kwargs['chunks']
        model = kwargs['model']
        summary = kwargs['summary']
        term_lists = kwargs['term_lists']

        # Validate the provided arguments
        if not isinstance(chunks, list):
            logger.error(f"Chunks must be a list")
            raise TypeError("Chunks must be a list")
        if not all(isinstance(chunk, Chunk) for chunk in chunks):
            logger.error(f"Chunks must be a list of Chunk objects")
            raise TypeError("Chunks must be a list of Chunk objects")
        if not isinstance(model, str):
            logger.error(f"Metadata model must be a string")
            raise TypeError("Metadata model must be a string")
        if model not in available_models:
            logger.error(f"Metadata model ({model}) must be a valid model. Available models: {', '.join(available_models)}")
            raise ValueError(f"Metadata model must be a valid model. Available models: {', '.join(available_models)}")
        if not isinstance(summary, str):
            logger.error(f"Summary must be a string")
            raise TypeError("Summary must be a string")

        task = Task(max_workers=4, retry_on_exceptions=(GPTTranslatorAPIRetryableException))

        translation = {}
        for chunk in chunks:
            sub_task = task.add_subtask(self._perform_translation_action, chunk=chunk, translation_model=model, summary=summary, term_lists=term_lists)
            translation[(sub_task, chunk.chunk_index)] = ""

        sorted_keys = sorted(translation.keys(), key=lambda x: x[1])
        translation = {key[0]: translation[key] for key in sorted_keys}

        results = task.run_subtasks()

        logger.info("Translating sub chapter complete.") 
        logger.info(f"Results: {results}")

        for task_id in translation:
            result = results[task_id]
            if isinstance(result, Exception):
                raise result
            translation[task_id] = result

        return "\n\n".join(translation.values())
    
    def _get_sub_chapter_context(self, novel: Novel, sub_chapter: SubChapter) -> tuple[SubChapter, SubChapter]:
        logger.debug(f"Getting sub chapter context.")
        
        # Validate the provided arguments
        if not isinstance(novel, Novel):
            logger.error(f"Novel must be a Novel object")
            raise TypeError("Novel must be a Novel object")
        if not isinstance(sub_chapter, SubChapter):
            logger.error(f"Sub chapter must be a SubChapter object")
            raise TypeError("Sub chapter must be a SubChapter object")
        
        # Check if the chapter and sub chapter indices are valid
        try:
            chapter = next(i_chapter for i_chapter in novel.chapters if i_chapter.chapter_index == sub_chapter.chapter_index)
            # sub_chapter = next(sub_chapter for sub_chapter in chapter.sub_chapters if sub_chapter.sub_chapter_index == sub_chapter.sub_chapter_index)
            # # chapter = novel.chapters[chapter_index]
            # # sub_chapter = chapter.sub_chapters[sub_chapter_index]
        except IndexError:
            logger.error(f"Invalid chapter or sub chapter index")
            raise GPTTranslatorException("Invalid chapter or sub chapter index")

        # Get the previous sub chapters
        if sub_chapter.sub_chapter_index == 1:
            if sub_chapter.chapter_index == 1:
                # If the sub chapter is the first sub chapter of the first chapter, there is no previous sub chapter
                prev_sub_chapter = None
            else:
                # If the sub chapter is the first sub chapter of a chapter, the previous sub chapter is the last sub chapter of the previous chapter
                try:
                    prev_sub_chapter = next(i_chapter for i_chapter in novel.chapters if i_chapter.chapter_index == sub_chapter.chapter_index - 1).sub_chapters[-1]
                except StopIteration:
                    prev_sub_chapter = None
                # prev_sub_chapter = novel.chapters[sub_chapter.chapter_index - 1].sub_chapters[-1]
        else:
            # If the sub chapter is not the first sub chapter of a chapter, the previous sub chapter is the previous sub chapter of the same chapter
            try:
                prev_sub_chapter = next(i_sub_chapter for i_sub_chapter in chapter.sub_chapters if i_sub_chapter.sub_chapter_index == sub_chapter.sub_chapter_index - 1)
            except StopIteration:
                prev_sub_chapter = None
            # prev_sub_chapter = chapter.sub_chapters[sub_chapter_index - 1]

        # Get the next sub chapters
        if sub_chapter.sub_chapter_index == max(i_sub_chapter.sub_chapter_index for i_sub_chapter in chapter.sub_chapters):
            if sub_chapter.chapter_index == max(i_chapter.chapter_index for i_chapter in novel.chapters):
                # If the sub chapter is the last sub chapter of the last chapter, there is no next sub chapter
                next_sub_chapter = None
            else:
                # If the sub chapter is the last sub chapter of a chapter, the next sub chapter is the first sub chapter of the next chapter
                try:
                    next_sub_chapter = next(i_chapter for i_chapter in novel.chapters if i_chapter.chapter_index == sub_chapter.chapter_index + 1).sub_chapters[0]
                except StopIteration:
                    next_sub_chapter = None
                # next_sub_chapter = novel.chapters[chapter_index + 1].sub_chapters[0]
        else:
            # If the sub chapter is not the last sub chapter of a chapter, the next sub chapter is the next sub chapter of the same chapter
            try:
                next_sub_chapter = next(i_sub_chapter for i_sub_chapter in chapter.sub_chapters if i_sub_chapter.sub_chapter_index == sub_chapter.sub_chapter_index + 1)
            except StopIteration:
                next_sub_chapter = None
            # next_sub_chapter = chapter.sub_chapters[sub_chapter_index + 1]

        logger.debug(f"Previous sub chapter: {prev_sub_chapter}, Next sub chapter: {next_sub_chapter}")
        return prev_sub_chapter, next_sub_chapter
    
    def translate_novel_metadata(self, novel: Novel) -> list[Exception]:
        logger.debug(f"Translating novel metadata.")

        # Validate the provided arguments
        if not isinstance(novel, Novel):
            logger.error(f"Novel must be a Novel object")
            raise TypeError("Novel must be a Novel object")
        
        # Get the metadata model
        model = self._get_api_model(self._metadata_models[0])['name']

        result = self._perform_novel_metadata_action(novel=novel, metadata_model=model)

        logger.debug("Translating novel metadata completed.")
        logger.debug(f"Result: {result}")
        
        if isinstance(result, Exception):
            return [result]

        return []
    
    def translate_sub_chapters_metadata(self, novel: Novel, targets: dict[str, list[str]]) -> list[Exception]:
        logger.debug(f"Translating sub chapters metadata.")

        # Validate parameters
        if not isinstance(novel, Novel):
            logger.error(f"Novel must be a Novel object")
            raise TypeError("Novel must be a Novel object")
        if not isinstance(targets, dict):
            logger.error(f"Targets must be a dictionary")
            raise TypeError("Targets must be a dictionary")
        if not all(isinstance(key, str) for key in targets.keys()):
            logger.error(f"Chapter numbers must be strings")
            raise TypeError("Chapter numbers must be strings")
        if not all(key.isdigit() for key in targets.keys()):
            logger.error(f"Chapter numbers must be digits as strings")
            raise TypeError("Chapter numbers must be digits as strings")
        if not all(isinstance(value, list) for value in targets.values()):
            logger.error(f"Sub chapter numbers must be lists")
            raise TypeError("Sub chapter numbers must be lists")
        if not all(isinstance(item, str) for value in targets.values() for item in value):
            logger.error(f"Sub chapter numbers must be strings")
            raise TypeError("Sub chapter numbers must be strings")
        if not all(item.isdigit() for value in targets.values() for item in value):
            logger.error(f"Sub chapter numbers must be digits as strings")
            raise TypeError("Sub chapter numbers must be digits as strings")
        
        # Get the metadata model
        model = self._get_api_model(self._metadata_models[0])['name']

        result = None
        
        sub_chapters = get_targeted_sub_chapters(novel, targets)

        # Translate the metadata
        result = self._perform_chapters_metadata_action(novel=novel, sub_chapters=sub_chapters, metadata_model=model)

        logger.debug("Translating sub chapters metadata complete.")
        logger.debug(f"Result: {result}")
        
        if isinstance(result, Exception):
            return [result]

        return []

    def summarize_sub_chapters(self, novel: Novel, targets: dict[str, list[str]]) -> list[Exception]:
        logger.debug(f"Summarizing sub chapters.")

        # Validate parameters
        if not isinstance(novel, Novel):
            logger.error(f"Novel must be a Novel object")
            raise TypeError("Novel must be a Novel object")
        if not isinstance(targets, dict):
            logger.error(f"Targets must be a dictionary")
            raise TypeError("Targets must be a dictionary")
        if not all(isinstance(key, str) for key in targets.keys()):
            logger.error(f"Chapter numbers must be strings")
            raise TypeError("Chapter numbers must be strings")
        if not all(key.isdigit() for key in targets.keys()):
            logger.error(f"Chapter numbers must be digits as strings")
            raise TypeError("Chapter numbers must be digits as strings")
        if not all(isinstance(value, list) for value in targets.values()):
            logger.error(f"Sub chapter numbers must be lists")
            raise TypeError("Sub chapter numbers must be lists")
        if not all(isinstance(item, str) for value in targets.values() for item in value):
            logger.error(f"Sub chapter numbers must be strings")
            raise TypeError("Sub chapter numbers must be strings")
        if not all(item.isdigit() for value in targets.values() for item in value):
            logger.error(f"Sub chapter numbers must be digits as strings")
            raise TypeError("Sub chapter numbers must be digits as strings")
        
        sub_chapters = get_targeted_sub_chapters(novel, targets)

        tasks = {}

        if all(self._target_language in sub_chapter.summary for sub_chapter in sub_chapters):
            logger.debug(f"All sub chapters are already summarized.")
            return []

        task = Task(max_workers=4, retry_on_exceptions=(GPTTranslatorAPIRetryableException))

        # Summarize the sub chapters
        for sub_chapter in sub_chapters:
            if self._target_language in sub_chapter.summary:
                continue

            prev_sub_chapter, next_sub_chapter = self._get_sub_chapter_context(novel, sub_chapter)

            # Get the previous and next lines and summary
            prev_line = prev_sub_chapter.contents.splitlines()[-1] if prev_sub_chapter and prev_sub_chapter.contents else ""
            next_line = next_sub_chapter.contents.splitlines()[0] if next_sub_chapter and next_sub_chapter.contents else ""

            # Calculate the optimal configuration for the sub chapter
            line_token_counts = self._calculate_line_token_counts(sub_chapter.contents)
            _, _, summary_division, _, _, summary_model = self._greedy_find_max_optimal_configuration(line_token_counts)

            # Split the text into chunks for the terms sheet API
            terms_chunks = self._split_text_into_chunks(sub_chapter.contents, summary_division, line_token_counts)
            terms_chunks_objects = list()
            for i, chunk in enumerate(terms_chunks):
                chunk_prev_line = prev_line if i == 0 else terms_chunks[i - 1].splitlines()[-1]
                chunk_next_line = next_line if i == len(terms_chunks) - 1 else terms_chunks[i + 1].splitlines()[0]

                terms_chunks_objects.append(Chunk(
                    novel.novel_code,
                    sub_chapter.chapter_index,
                    sub_chapter.sub_chapter_index,
                    i,
                    chunk,
                    chunk_prev_line,
                    chunk_next_line))
            
            sub_task = task.add_subtask(self._summarize_sub_chapter, chunks=terms_chunks_objects, model=summary_model)
            tasks[(sub_chapter.chapter_index, sub_chapter.sub_chapter_index)] = sub_task

        results = task.run_subtasks()

        logger.debug(f"Summarizing sub chapters complete.")
        logger.debug(f"Results: {results}")
        
        exceptions = []
        for (chapter_index, sub_chapter_index), sub_task in tasks.items():
            logger.debug(f"Processing sub task {sub_task} for chapter {chapter_index} sub chapter {sub_chapter_index}")
            result = results[sub_task]
            logger.debug(f"Result: {result}")
            if isinstance(result, Exception):
                exceptions.append(result)
            else: 
                returned_chapter_index, returned_sub_chapter_index, summary = result
                active_chapter = novel.get_chapter(returned_chapter_index)
                active_sub_chapter = active_chapter.get_sub_chapter(returned_sub_chapter_index)
                new_summary = active_sub_chapter.summary.copy()
                new_summary[self._target_language] = summary
                active_sub_chapter.summary = new_summary
                logger.debug(f"Assigned summary to chapter {chapter_index} sub chapter {sub_chapter_index}")
                logger.debug(f"Current summary: {active_sub_chapter.summary[self._target_language]}")

        return exceptions
    
    def gather_terms_for_sub_chapters(self, novel: Novel, targets: dict[str, list[str]]) -> list[Exception]:
        logger.debug(f"Gathering terms for sub chapters.")

        # Validate parameters
        if not isinstance(novel, Novel):
            logger.error(f"Novel must be a Novel object")
            raise TypeError("Novel must be a Novel object")
        if not isinstance(targets, dict):
            logger.error(f"Targets must be a dictionary")
            raise TypeError("Targets must be a dictionary")
        if not all(isinstance(key, str) for key in targets.keys()):
            logger.error(f"Chapter numbers must be strings")
            raise TypeError("Chapter numbers must be strings")
        if not all(key.isdigit() for key in targets.keys()):
            logger.error(f"Chapter numbers must be digits as strings")
            raise TypeError("Chapter numbers must be digits as strings")
        if not all(isinstance(value, list) for value in targets.values()):
            logger.error(f"Sub chapter numbers must be lists")
            raise TypeError("Sub chapter numbers must be lists")
        if not all(isinstance(item, str) for value in targets.values() for item in value):
            logger.error(f"Sub chapter numbers must be strings")
            raise TypeError("Sub chapter numbers must be strings")
        if not all(item.isdigit() for value in targets.values() for item in value):
            logger.error(f"Sub chapter numbers must be digits as strings")
            raise TypeError("Sub chapter numbers must be digits as strings")
        
        sub_chapters = get_targeted_sub_chapters(novel, targets)

        tasks = {}

        task = Task(max_workers=4, retry_on_exceptions=(GPTTranslatorAPIRetryableException))

        # Summarize the sub chapters
        for sub_chapter in sub_chapters:
            prev_sub_chapter, next_sub_chapter = self._get_sub_chapter_context(novel, sub_chapter)

            # Get the previous and next lines and summary
            prev_line = prev_sub_chapter.contents.splitlines()[-1] if prev_sub_chapter and prev_sub_chapter.contents else ""
            next_line = next_sub_chapter.contents.splitlines()[0] if next_sub_chapter and next_sub_chapter.contents else ""

            # Calculate the optimal configuration for the sub chapter
            line_token_counts = self._calculate_line_token_counts(sub_chapter.contents)
            terms_division, _, _, terms_model, _, _ = self._greedy_find_max_optimal_configuration(line_token_counts)

            # Split the text into chunks for the terms sheet API
            terms_chunks = self._split_text_into_chunks(sub_chapter.contents, terms_division, line_token_counts)
            terms_chunks_objects = list()
            for i, chunk in enumerate(terms_chunks):
                chunk_prev_line = prev_line if i == 0 else terms_chunks[i - 1].splitlines()[-1]
                chunk_next_line = next_line if i == len(terms_chunks) - 1 else terms_chunks[i + 1].splitlines()[0]

                terms_chunks_objects.append(Chunk(
                    novel.novel_code,
                    sub_chapter.chapter_index,
                    sub_chapter.sub_chapter_index,
                    i,
                    chunk,
                    chunk_prev_line,
                    chunk_next_line))
            
            sub_task = task.add_subtask(self._gather_terms_for_sub_chapter, chunks=terms_chunks_objects, model=terms_model)
            tasks[(sub_chapter.chapter_index, sub_chapter.sub_chapter_index)] = sub_task

        results = task.run_subtasks()

        logger.debug(f"Gathering terms for sub chapters complete.")
        logger.debug(f"{results}")

        terms = ""
        exceptions = []
        for _, value in tasks.items():
            result = results[value]
            if isinstance(result, Exception):
                exceptions.append(result)
            else:
                terms += result + "\n\n"

        if not novel.terms_sheet:
            novel.terms_sheet = TermSheet(novel.novel_origin, novel.novel_code)
        novel.terms_sheet.process_new_terms(terms)

        return exceptions
    
    def translate_sub_chapters(self, novel: Novel, targets: dict[str, list[str]]) -> list[Exception]:
        logger.debug(f"Translating sub chapters.")

        # Validate parameters
        if not isinstance(novel, Novel):
            logger.error(f"Novel must be a Novel object")
            raise TypeError("Novel must be a Novel object")
        if not isinstance(targets, dict):
            logger.error(f"Targets must be a dictionary")
            raise TypeError("Targets must be a dictionary")
        if not all(isinstance(key, str) for key in targets.keys()):
            logger.error(f"Chapter numbers must be strings")
            raise TypeError("Chapter numbers must be strings")
        if not all(key.isdigit() for key in targets.keys()):
            logger.error(f"Chapter numbers must be digits as strings")
            raise TypeError("Chapter numbers must be digits as strings")
        if not all(isinstance(value, list) for value in targets.values()):
            logger.error(f"Sub chapter numbers must be lists")
            raise TypeError("Sub chapter numbers must be lists")
        if not all(isinstance(item, str) for value in targets.values() for item in value):
            logger.error(f"Sub chapter numbers must be strings")
            raise TypeError("Sub chapter numbers must be strings")
        if not all(item.isdigit() for value in targets.values() for item in value):
            logger.error(f"Sub chapter numbers must be digits as strings")
            raise TypeError("Sub chapter numbers must be digits as strings")
        
        sub_chapters = get_targeted_sub_chapters(novel, targets)

        tasks = {}

        task = Task(max_workers=4, retry_on_exceptions=(GPTTranslatorAPIRetryableException))

        # Summarize the sub chapters
        for sub_chapter in sub_chapters:
            if self._target_language in sub_chapter.translation:
                continue

            prev_sub_chapter, next_sub_chapter = self._get_sub_chapter_context(novel, sub_chapter)

            # Get the previous and next lines and summary
            prev_line = prev_sub_chapter.contents.splitlines()[-1] if prev_sub_chapter and prev_sub_chapter.contents else ""
            next_line = next_sub_chapter.contents.splitlines()[0] if next_sub_chapter and next_sub_chapter.contents else ""

            # Calculate the optimal configuration for the sub chapter
            line_token_counts = self._calculate_line_token_counts(sub_chapter.contents)
            _, translate_division, _, _, translate_model, _ = self._greedy_find_max_optimal_configuration(line_token_counts)

            # Split the text into chunks for the terms sheet API
            terms_chunks = self._split_text_into_chunks(sub_chapter.contents, translate_division, line_token_counts)
            terms_chunks_objects = list()
            for i, chunk in enumerate(terms_chunks):
                chunk_prev_line = prev_line if i == 0 else terms_chunks[i - 1].splitlines()[-1]
                chunk_next_line = next_line if i == len(terms_chunks) - 1 else terms_chunks[i + 1].splitlines()[0]

                terms_chunks_objects.append(Chunk(
                    novel.novel_code,
                    sub_chapter.chapter_index,
                    sub_chapter.sub_chapter_index,
                    i,
                    chunk,
                    chunk_prev_line,
                    chunk_next_line))
                
            sub_task = task.add_subtask(self._translate_sub_chapter, chunks=terms_chunks_objects, model=translate_model, summary=sub_chapter.summary[self._target_language], term_lists=novel.terms_sheet)
            tasks[(sub_chapter.chapter_index, sub_chapter.sub_chapter_index)] = sub_task

        results = task.run_subtasks()

        logger.debug(f"Translating sub chapters complete.")
        logger.debug(f"{results}")

        translation = ""
        exceptions = []
        for (chapter_index, sub_chapter_index), sub_task in tasks.items():
            result = results[sub_task]
            if isinstance(result, Exception):
                exceptions.append(result)
            else:
                active_chapter = novel.get_chapter(chapter_index)
                active_sub_chapter = active_chapter.get_sub_chapter(sub_chapter_index)
                new_translation = active_sub_chapter.translation.copy()
                new_translation[self._target_language] = result
                active_sub_chapter.translation = new_translation
                logger.debug(f"Assigned translation to chapter {chapter_index} sub chapter {sub_chapter_index}")
                logger.debug(f"Translation: {result}")

        return exceptions

@singleton
class GPTTranslatorSingleton(GPTTranslator):
    def __init__(self) -> None:
        logger.info("Initializing GPTTranslator")
        cf = Config()
        available_models = cf.data.config.openai.models
        terms_models = cf.data.config.translator.api.terms_list.models
        translation_models = cf.data.config.translator.api.translation.models
        summary_models = cf.data.config.translator.api.summary.models
        metadata_models = cf.data.config.translator.api.metadata.models
        original_language = ""
        target_language = cf.data.config.translator.target_language
        self._initialize(available_models, terms_models, translation_models, summary_models, metadata_models, original_language, target_language)
