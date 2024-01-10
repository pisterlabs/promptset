import openai
openai.api_key = "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"
MODEL = "text-davinci-003"
MAX_TOKENS = 2000
model_to_api = {
    "text-davinci-003": "Completion",
    "gpt-3.5": "Completion",
    "gpt-4": "ChatCompletion"  
}
QUESTION_PATH = "/root/.leetcode/code/"
LC_PATH = "./data/lc/"
GPT_PATH = "./data/gpt/"
PROMPT_PATH = "./data/prompt/"
OJ_PATH = "./data/oj/"
MEM_PATH = "./data/mem/"
UNIT_TEST_PATH = "./data/unit_test/"
UT_PROMPT_PATH = "./data/ut_prompt/"

PREFIX = "### Instruction: You are a helpful AI Assistant. Please provide python solution based on the user's instructions,  \\\
         please only return python code. Please consider optimized time and space complexity.  ### Input: "
PROMPT = "Return the solution using the following class definition:\n"
RESPONSE = "### Response:"
UNIT_TEST = "### Instruction: You are a helpful AI Assistant. Please provide 10 valid test cases for the following program based on the user's instructions,  \\\
         the output should be equal to the actual return value of the input, please only return input and output. Please cover corner cases and hard cases.  ### Input: "
PROMPT_UT = "Return 10 test cases using the following class definition:\n"
        