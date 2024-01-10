import unittest
import os
import tempfile

import openai

from utils import file_reader
from index_resumes import main, parse_aim_resumes, embed_example
from unittest.mock import patch, Mock
import dotenv

dotenv.load_dotenv()

class TestAIMProcessing(unittest.TestCase):

    def setUp(self):
        # Temporary directory to mock the 'profiles' structure
        self.temp_dir = tempfile.TemporaryDirectory()
        self.profile_path = os.path.join(self.temp_dir.name, 'profiles')
        os.makedirs(self.profile_path)
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        # Create sample AIM profile and resume
        self.sample_aim_content = (
            "SUMMARY\nSample summary content\n"
            "SKILLS AND TECHNOLOGIES\nSample skills content\n"
            "PROFESSIONAL EXPERIENCE\nSample experience content\n"
            "EDUCATION\nSample education content"
        )
        self.sample_resume_content = "This is a sample resume content."

        # Mock directory for a user
        self.user_dir = os.path.join(self.profile_path, 'user1')
        os.makedirs(self.user_dir)
        with open(os.path.join(self.user_dir, 'AIM P profile.docx'), 'w') as f:
            f.write(self.sample_aim_content)
        with open(os.path.join(self.user_dir, 'resume.txt'), 'w') as f:
            f.write(self.sample_resume_content)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_file_reader(self):
        with open(os.path.join(self.user_dir, 'resume.txt'), 'r') as f:
            content = file_reader(f)
        self.assertEqual(content, self.sample_resume_content)
        # Add similar checks for .doc, .pdf etc.

    def test_parse_aim_resumes(self):
        summary, skills, experience, education = parse_aim_resumes(self.sample_aim_content)
        self.assertEqual(summary.strip(), "Sample summary content")
        self.assertEqual(skills.strip(), "Sample skills content")
        self.assertEqual(experience.strip(), "Sample experience content")
        self.assertEqual(education.strip(), "Sample education content")

    @patch('index_resumes.OpenAIEmbeddings')
    @patch('index_resumes.FAISS')
    def test_main_function(self, MockFAISS, MockEmbeddings):
        # We're patching external libraries to prevent real interactions
        mock_embedder_instance = Mock()
        mock_embedder_instance.embed_query.return_value = "mock_embedding"
        MockEmbeddings.return_value = mock_embedder_instance

        mock_faiss_instance = Mock()
        MockFAISS.return_value = mock_faiss_instance

        main('../profiles')  # As the profiles dir is created in setUp, it'll work on that




if __name__ == '__main__':
    unittest.main()
