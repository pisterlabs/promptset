import unittest
from unittest.mock import patch
from recruit_flow_ai.settings import OpenaiSettings

class TestSettings(unittest.TestCase):
    """
    This class is for unit testing the Settings class.
    It tests the api_key attribute and the Config inner class.
    """
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'})
    def test_api_key(self):
        settings = OpenaiSettings()
        self.assertEqual(settings.api_key.get_secret_value(), 'test_key')

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'})
    def test_config(self):
        settings = OpenaiSettings()
        self.assertEqual(settings.Config.env_file, '.env')
        self.assertEqual(settings.Config.env_prefix, 'OPENAI_')
        self.assertEqual(settings.Config.case_sensitive, False)

if __name__ == '__main__':
    unittest.main()
