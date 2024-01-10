from unittest import TestCase
from unittest.mock import patch

from sdb.ai_utils import get_ai_summary_for_arabic_text


class GetAISummaryForArabicTextTestCase(TestCase):
    """Tests for ai_utils.py"""

    @patch("openai.ChatCompletion.create")
    def test_get_ai_summary_for_arabic_text(self, mock_openai):
        """Does get_ai_summary_for_arabic_text run without error?"""

        mock_openai.return_value = {
            "choices": [{"message": {"content": "Summarized text"}}]
        }

        """Should run without throwing error"""
        result = get_ai_summary_for_arabic_text("اختبار")

        """Should return content of response from OpenAI API call"""
        self.assertEqual(result, "Summarized text")
