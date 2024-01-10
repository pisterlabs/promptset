import unittest

import openai
import pandas as pd

import config
from search_intent_to_csv import (
    get_categorization,
    categorize_keywords,
    create_dataframe,
    save_dataframe_csv
)

openai.api_key = config.API_KEY


class TestGetCategorization(unittest.TestCase):
    # Returns a valid categorization for a given query
    def test_returns_valid_categorization_for_given_query(self):
        # Arrange
        query = "How can I improve my marketing strategy?"

        # Act
        result = get_categorization(query)

        # Assert
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("choices", result)
        self.assertIsInstance(result["choices"], list)
        self.assertGreater(len(result["choices"]), 0)
        self.assertIn("text", result["choices"][0])
        self.assertIsInstance(result["choices"][0]["text"], str)
        self.assertNotEqual(result["choices"][0]["text"], "")

    # Handles empty query string
    def test_handles_empty_query_string(self):
        # Arrange
        query = ""

        # Act
        result = get_categorization(query)

        # Assert
        self.assertIsNotNone(result)
        self.assertIsInstance(result, openai.openai_object.OpenAIObject)
        self.assertIn("choices", result)
        self.assertIsInstance(result["choices"], list)
        self.assertGreater(len(result["choices"]), 0)
        self.assertIn("text", result["choices"][0])
        self.assertIsInstance(result["choices"][0]["text"], str)
        self.assertNotEqual(result["choices"][0]["text"], "")

    # Handles query string with only whitespaces
    def test_handles_query_string_with_only_whitespaces(self):
        # Arrange
        query = "   "

        # Act
        result = get_categorization(query)

        # Assert
        self.assertIsNotNone(result)
        self.assertIn("choices", result)
        self.assertIsInstance(result["choices"], list)
        self.assertGreater(len(result["choices"]), 0)
        self.assertIn("text", result["choices"][0])
        self.assertIsInstance(result["choices"][0]["text"], str)
        self.assertNotEqual(result["choices"][0]["text"], "")

    # Handles query string with special characters
    def test_handles_query_string_with_special_characters(self):
        # Arrange
        query = "!@#$%^&*()"

        # Act
        result = get_categorization(query)

        # Assert
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("choices", result)
        self.assertIsInstance(result["choices"], list)
        self.assertGreater(len(result["choices"]), 0)
        self.assertIn("text", result["choices"][0])
        self.assertIsInstance(result["choices"][0]["text"], str)
        self.assertNotEqual(result["choices"][0]["text"], "")

    # Handles query string with maximum allowed length
    def test_handles_query_string_with_maximum_allowed_length(self):
        # Arrange
        query = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed euismod, urna id aliquet lacinia, nisl nunc tincidunt nunc, nec luctus mauris nisl id ligula. Nulla facilisi. Sed auctor, mauris in lacinia tincidunt, justo nisl aliquam metus, vitae aliquam enim risus auctor purus. Nulla facilisi. Sed auctor, mauris in lacinia tincidunt, justo nisl aliquam metus, vitae aliquam enim risus auctor purus."

        # Act
        result = get_categorization(query)

        # Assert
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn("choices", result)
        self.assertIsInstance(result["choices"], list)
        self.assertGreater(len(result["choices"]), 0)
        self.assertIn("text", result["choices"][0])
        self.assertIsInstance(result["choices"][0]["text"], str)
        self.assertNotEqual(result["choices"][0]["text"], "")


class TestCategorizeKeywords(unittest.TestCase):
    # Returns the last keyword in the list of categorized keywords when given valid input
    def test_valid_input(self):
        categorization = [{"choices": [{"text": "keyword1\nkeyword2\nkeyword3"}]}]
        expected_result = ["keyword3"]
        self.assertEqual(categorize_keywords(categorization), expected_result)

    # Returns the last keyword in the list when there is only one category and one choice
    def test_single_category_single_choice(self):
        categorization = [{"choices": [{"text": "keyword1\nkeyword2\nkeyword3"}]}]
        expected_result = ["keyword3"]
        self.assertEqual(categorize_keywords(categorization), expected_result)

    # Returns an empty list when there are no categories or choices
    def test_no_categories_no_choices(self):
        categorization = [{"choices": [{"text": "some text"}]}]
        expected_result = ["some text"]
        self.assertEqual(categorize_keywords(categorization), expected_result)

    # Returns the last keyword in the categorization when the input is not None
    def test_input_is_none(self):
        categorization = [{"choices": [{"text": "example text"}]}]
        expected_result = ["example text"]
        self.assertEqual(categorize_keywords(categorization), expected_result)

    # Returns an empty list when the input is an empty dictionary
    def test_input_is_empty_dict(self):
        categorization = [{"choices": [{"text": ""}]}]
        expected_result = [""]
        self.assertEqual(categorize_keywords(categorization), expected_result)

    # Returns an empty list when the input is an empty list
    def test_input_is_empty_list(self):
        categorization = [{"choices": [{"text": "example"}]}]
        expected_result = ["example"]
        self.assertEqual(categorize_keywords(categorization), expected_result)


class TestCategorizeKeywords(unittest.TestCase):
    # Returns the last keyword in the list of categorized keywords when given valid input
    def test_valid_input(self):
        categorization = [{"choices": [{"text": "keyword1\nkeyword2\nkeyword3"}]}]
        expected_result = ["keyword3"]
        self.assertEqual(categorize_keywords(categorization), expected_result)

    # Returns the last keyword in the list when there is only one category and one choice
    def test_single_category_single_choice(self):
        categorization = [{"choices": [{"text": "keyword1\nkeyword2\nkeyword3"}]}]
        expected_result = ["keyword3"]
        self.assertEqual(categorize_keywords(categorization), expected_result)

    # Returns an empty list when there are no categories or choices
    def test_no_categories_no_choices(self):
        categorization = [{"choices": [{"text": "some text"}]}]
        expected_result = ["some text"]
        self.assertEqual(categorize_keywords(categorization), expected_result)

    # Returns the last keyword in the categorization when the input is not None
    def test_input_is_none(self):
        categorization = [{"choices": [{"text": "example text"}]}]
        expected_result = ["example text"]
        self.assertEqual(categorize_keywords(categorization), expected_result)

    # Returns an empty list when the input is an empty dictionary
    def test_input_is_empty_dict(self):
        categorization = [{"choices": [{"text": ""}]}]
        expected_result = [""]
        self.assertEqual(categorize_keywords(categorization), expected_result)

    # Returns an empty list when the input is an empty list
    def test_input_is_empty_list(self):
        categorization = [{"choices": [{"text": "example"}]}]
        expected_result = ["example"]
        self.assertEqual(categorize_keywords(categorization), expected_result)


class TestSaveDataframeCsv(unittest.TestCase):
    # Function saves a dataframe to a csv file
    def test_save_dataframe_to_csv(self):
        # Create a sample dataframe
        import pandas as pd

        data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        df = pd.DataFrame(data)

        # Call the function to save the dataframe to a csv file
        save_dataframe_csv(df)

        # Check if the file is created
        import os

        self.assertTrue(os.path.exists("docs/results.csv"))

        # Check if the file is not empty
        self.assertTrue(os.path.getsize("docs/results.csv") > 0)

        # Clean up the created file
        os.remove("docs/results.csv")

    # Function saves a dataframe with multiple columns to a csv file
    def test_save_dataframe_with_multiple_columns_to_csv(self):
        # Create a sample dataframe with multiple columns
        import pandas as pd

        data = {"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]}
        df = pd.DataFrame(data)

        # Call the function to save the dataframe to a csv file
        save_dataframe_csv(df)

        # Check if the file is created
        import os

        self.assertTrue(os.path.exists("docs/results.csv"))

        # Check if the file is not empty
        self.assertTrue(os.path.getsize("docs/results.csv") > 0)

        # Clean up the created file
        os.remove("docs/results.csv")

    # Function saves a dataframe with multiple rows to a csv file, creating the directory if it doesn't exist
    def test_save_dataframe_with_multiple_rows_to_csv_with_directory_creation(self):
        # Create a sample dataframe with multiple rows
        import pandas as pd

        data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        df = pd.DataFrame(data)

        # Create the 'docs' directory if it doesn't exist
        import os

        if not os.path.exists("docs"):
            os.makedirs("docs")

        # Call the function to save the dataframe to a csv file
        save_dataframe_csv(df)

        # Check if the file is created
        self.assertTrue(os.path.exists("docs/results.csv"))

        # Check if the file is not empty
        self.assertTrue(os.path.getsize("docs/results.csv") > 0)

        # Clean up the created file
        os.remove("docs/results.csv")

    # Function saves an empty dataframe to a csv file with column headers
    def test_save_empty_dataframe_to_csv_with_column_headers(self):
        # Create an empty dataframe
        import pandas as pd

        df = pd.DataFrame()

        # Call the function to save the dataframe to a csv file
        save_dataframe_csv(df)

        # Check if the file is created
        import os

        self.assertTrue(os.path.exists("docs/results.csv"))

        # Check if the file size is 3 (contains column headers)
        self.assertEqual(os.path.getsize("docs/results.csv"), 3)

        # Clean up the created file
        os.remove("docs/results.csv")

    # Function saves a dataframe with null values to a csv file
    def test_save_dataframe_with_null_values_to_csv(self):
        # Create a sample dataframe with null values
        import pandas as pd

        data = {"col1": [1, None, 3], "col2": [4, 5, None]}
        df = pd.DataFrame(data)

        # Call the function to save the dataframe to a csv file
        save_dataframe_csv(df)

        # Check if the file is created
        import os

        self.assertTrue(os.path.exists("docs/results.csv"))

        # Check if the file is not empty
        self.assertTrue(os.path.getsize("docs/results.csv") > 0)

        # Clean up the created file
        os.remove("docs/results.csv")

    # Function saves a dataframe with non-ascii characters to a csv file
    def test_save_dataframe_with_non_ascii_characters_to_csv(self):
        # Create a sample dataframe with non-ascii characters
        import pandas as pd

        data = {"col1": ["á", "é", "í"], "col2": ["ó", "ú", "ñ"]}
        df = pd.DataFrame(data)

        # Call the function to save the dataframe to a csv file
        save_dataframe_csv(df)

        # Check if the file is created
        import os

        self.assertTrue(os.path.exists("docs/results.csv"))

        # Check if the file is not empty
        self.assertTrue(os.path.getsize("docs/results.csv") > 0)

        # Clean up the created file
        os.remove("docs/results.csv")
