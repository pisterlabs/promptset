import os
import sys
import unittest
import tiktoken

# Allow this file to import from the parent dir
this_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(this_dir)
sys.path.append(parent_dir)

from openai_utils import tokenize, generate_prompts

def validate_token_size(segments, model, max_tokens):
    for segment in segments:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(segment)
        print("Num tokens:", len(tokens))
        if len(tokens) > max_tokens:
            return False
    return True

class TestTokenizer(unittest.TestCase):

    
    """
    def test_empty_string(self):
        model = 'gpt-4'
        max_tokens = 10
        prompts = tokenize("", model, max_tokens)
        self.assertEqual(prompts, [])
        self.assertTrue(self.validate_prompt_sizes(prompts, model, max_tokens))

    def test_last_token_is_delimiter(self):
        model = 'gpt-4'
        max_tokens = 5
        prompts = tokenize("This is a test.", model, max_tokens)
        self.assertEqual(prompts, ['This is a test.'])
        self.assertTrue(self.validate_prompt_sizes(prompts, model, max_tokens))

    def test_last_token_is_not_delimiter(self):
        model = 'gpt-4'
        max_tokens = 5
        text = "Test. This is a really long sentence."
        prompts = tokenize(text, model, max_tokens)
        self.assertEqual(prompts, ['Test.', ' This is a really long', ' sentence.'])
        self.assertTrue(self.validate_prompt_sizes(prompts, model, max_tokens))
        # print("prompts:", prompts)

    def test_single_token_no_delimiter(self):
        pass

    def test_multiple_tokens_no_delimiter(self):
        pass

    def test_korean_0(self):
        with open(this_dir + "/inputs/memorize-ch-1.txt", "r") as f:
            text = f.read()
        model = 'gpt-4'
        max_tokens = 4000
        prompts = tokenize(text, model, max_tokens)
        print("Here!")
        self.assertTrue(self.validate_prompt_sizes(prompts, model, max_tokens))
        
        with open(this_dir + "/outputs/memorize-ch-1.txt", "w") as f:
            # f.writelines(prompts)
            f.write(prompts[0])
    """
    # def test_korean_1(self):
    #     with open(this_dir + "/inputs/memorize-ch-1-sample-1.txt", "r") as f:
    #         text = f.read()
    #     model = 'gpt-4'
    #     max_tokens = 4000
    #     prompts = tokenize(text, model, max_tokens)
    #     self.assertEqual(len(prompts), 1)
        
    #     with open(this_dir + "/outputs/memorize-ch-1-sample-1.txt", "w") as f:
    #         # f.writelines(prompts)
    #         f.write(prompts[0])
    

    # def test_ends_with_ending_mark_and_quote(self):
    #     model = 'gpt-4'
    #     max_tokens = 10
    #     text = '''Hi.\n"Hi."'''
    #     prompts = tokenize(text, model, max_tokens)
    #     self.assertEqual(len(prompts), 1)
    
        
    # def test_ends_with_ending_mark_and_quote(self):
    #     model = 'gpt-4'
    #     max_tokens = 1
    #     text = '''Hello.World!A?Sky."Is!"Blue?"One\nTwo.\nThree!\nFour?\n'''
    #     prompts = tokenize(text, model, max_tokens)
    #     self.assertEqual(len(prompts), 19)

    def test_single_segment_with_no_delimiter_at_end(self):
        """Test when the segment does not end with a delimiter.
        Should not split into multiple prompts. 
        Should return entire segment in single prompt."""
        pass

class TestPromptGenerator(unittest.TestCase):

    def test_prompt_size_less_than_token_limit(self):
        pass

    def test_large_text_0(self):
        prompt_token_limit = 2700
        with open(this_dir + "/inputs/memorize-ch-1.txt", "r") as f:
            text = f.read()
        
        prompts = generate_prompts("Translate this to english:\n", text, 'gpt-4', prompt_token_limit)
        with open(this_dir + "/outputs/test_large_text_0.txt", "w") as f:
            # f.writelines(prompts)
            f.write(prompts[0])

        self.assertTrue(validate_token_size(prompts, 'gpt-4', prompt_token_limit))

    def test_prompt_prefix_size(self):
        """Test that an error is thrown when the prompt prefix token size >= prompt token limit."""
        pass

    def test_count_tokens(self):
        with open(this_dir + '/inputs/count_tokens.txt', 'r') as f:
            text = f.read()
        encoding = tiktoken.encoding_for_model('gpt-4')
        tokens = encoding.encode(text)
        print("count tokens", len(tokens))

if __name__ == '__main__':
    unittest.main()
