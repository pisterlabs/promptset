import os
import unittest
from unittest.mock import patch, mock_open
from dotenv import load_dotenv
import openai
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, base_dir)
# pylint: disable=wrong-import-position
from main import OpenAIGenerator


class TestOpenAIGenerator(unittest.TestCase):
    """
    Test class for OpenAIGenerator
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up class method to run before each test cases.
        """

        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @patch('json.load')
    @patch('builtins.open', read_data='data')
    def setUp(self, _, mock_json):  # pylint: disable=arguments-differ
        """
        Set up method to run before each test cases.
        """

        mock_json.return_value = {
            "dynamic-prompting": True,
            "dynamic-prompting-examples": 3,
            "first-step-iterations": 1,
            "second-step-iterations": 1,
            "themes_dict": {"maths": ["general", "matrices", "derivatives"]}
        }
        self.generator = OpenAIGenerator()

    def tearDown(self):
        """
        Tear down method that does clean up after each test case has run.
        """

        self.generator = None

    def test_model(self):
        """
        Test the model method
        """

        prompt = "This is a mock prompt."
        self.generator.model(prompt)

        self.assertGreater(len(self.generator.response), 0)

    @patch('main.OpenAIGenerator.write_csv')
    @patch('main.OpenAIGenerator.model')
    @patch('main.Prompt.get_prompt')
    @patch('time.sleep')
    def test_generate_dataset(self, mock_sleep, mock_get_prompt, mock_model, mock_write_csv):
        """
        Test the generate_dataset method
        """

        mock_get_prompt.return_value = 'mock prompt'
        self.generator.generate_dataset(identifier=1, subject='maths')

        # Check if the methods were called with the correct arguments
        mock_get_prompt.assert_called_with(f='src/datasets/manual/manual-questions-maths.csv',
                                           subject='maths', identifier=1)
        mock_model.assert_called_with(prompt_content='mock prompt')
        mock_write_csv.assert_called_with(subject='maths')

        # Check if the methods were called the correct number of times
        self.assertEqual(mock_get_prompt.call_count, self.generator.config["first-step-iterations"])
        self.assertEqual(mock_model.call_count, self.generator.config["first-step-iterations"])
        self.assertEqual(mock_write_csv.call_count, self.generator.config["first-step-iterations"])
        self.assertEqual(mock_sleep.call_count, self.generator.config["first-step-iterations"])

    def test_merge_input_output(self):
        """
        Test the merge_input_output method
        """
        # Create a mock dataset
        mock_dataset = pd.DataFrame({
            'input': ['input1', 'input2', 'input3'],
            'output': ['output1', 'output2', 'output3']
        })

        # Call the method with the mock dataset
        result_dataset = self.generator.merge_input_output(mock_dataset)

        # Check if the 'text' column was correctly created
        expected_text_column = ['input1->: output1', 'input2->: output2', 'input3->: output3']
        self.assertListEqual(result_dataset['text'].tolist(), expected_text_column)

    @patch('builtins.open', new_callable=mock_open, read_data='data')
    @patch('pandas.read_csv')
    def test_add_manual_questions(self, mock_read_csv, mockopen):
        """
        Test the add_manual_questions method
        """
        # Create a mock dataset
        mock_dataset = pd.DataFrame({
            'Column1': ['instruction1', 'instruction2', 'instruction3'],
            'Column2': ['input1', 'input2', 'input3'],
            'Column3': ['topic1', 'topic2', 'topic3'],
            'Column4': ['subject1', 'subject2', 'subject3']
        })

        mock_read_csv.return_value = mock_dataset

        # Call the method with the mock dataset
        self.generator.add_manual_questions('mock_manual_questions.csv', 'mock_dataset.csv')

        # Check if the methods were called with the correct arguments
        mock_read_csv.assert_called_with('mock_manual_questions.csv', delimiter=";")
        mockopen.assert_called_with('mock_dataset.csv', "a", encoding="utf-8")

        # Check if the write method was called with the correct arguments
        mockopen().write.assert_any_call(
            "Tu es un analyseur de données charge d'aider les étudiants à trouver des "
            "ressources répond au mieux en format JSON.;input1;{\"topic\": topic1, \"subject\": "
            "subject1};\n"
        )
        mockopen().write.assert_any_call(
            "Tu es un analyseur de données charge d'aider les étudiants à trouver des "
            "ressources répond au mieux en format JSON.;input2;{\"topic\": topic2, \"subject\": "
            "subject2};\n"
        )
        mockopen().write.assert_any_call(
            "Tu es un analyseur de données charge d'aider les étudiants à trouver des "
            "ressources répond au mieux en format JSON.;input3;{\"topic\": topic3, \"subject\": "
            "subject3};\n"
        )

        # Check if the write method was called the correct number of times
        self.assertEqual(mockopen().write.call_count, mock_dataset.shape[0])


if __name__ == '__main__':
    unittest.main()
