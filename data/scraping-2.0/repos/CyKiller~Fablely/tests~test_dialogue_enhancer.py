import unittest
from fablely.dialogue_enhancer import DialogueEnhancer
from typing import Dict, Any
import langchain

class TestDialogueEnhancer(unittest.TestCase):
    ## Test Initialization
    def test_initialization(self):
        """
        Test if the DialogueEnhancer is initialized correctly.
        """
        dialogue_enhancer = DialogueEnhancer("Hello, world!")
        self.assertEqual(dialogue_enhancer.text, "Hello, world!")
        self.assertIsInstance(dialogue_enhancer.enhancer, langchain.DialogueEnhancer)

    ## Test enhance_dialogue
    def test_enhance_dialogue(self):
        """
        Test if the enhance_dialogue method works correctly.
        """
        dialogue_enhancer = DialogueEnhancer("Hello, world!")
        enhanced_text = dialogue_enhancer.enhance_dialogue()
        self.assertIsInstance(enhanced_text, Dict[str, Any])

    ## Test enhance_dialogue with empty string
    def test_enhance_dialogue_empty_string(self):
        """
        Test if the enhance_dialogue method handles an empty string correctly.
        """
        dialogue_enhancer = DialogueEnhancer("")
        enhanced_text = dialogue_enhancer.enhance_dialogue()
        self.assertIsInstance(enhanced_text, Dict[str, Any])

    ## Test enhance_dialogue with None
    def test_enhance_dialogue_none(self):
        """
        Test if the enhance_dialogue method handles None correctly.
        """
        with self.assertRaises(TypeError):
            dialogue_enhancer = DialogueEnhancer(None)
            enhanced_text = dialogue_enhancer.enhance_dialogue()

    ## Test enhance_dialogue with non-string
    def test_enhance_dialogue_non_string(self):
        """
        Test if the enhance_dialogue method handles non-string inputs correctly.
        """
        with self.assertRaises(TypeError):
            dialogue_enhancer = DialogueEnhancer(123)
            enhanced_text = dialogue_enhancer.enhance_dialogue()

if __name__ == '__main__':
    unittest.main()
