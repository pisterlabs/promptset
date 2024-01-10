#
# Driver program to submit paragraphs from SemEval 2021 - Task 8
# MeasEval to GPT-3 via the OpenAI API.
#

import argparse

import openai
from transformers import GPT2TokenizerFast

parser = argparse.ArgumentParser()

parser.add_argument("keyfilePath", help="Directory containing keyfiles to process")
parser.add_argument("keyfileName", help="file containing the paragraph keys to process")
parser.add_argument("paragraphDirectory", help="Directory containing the paragraphs to process")
parser.add_argument("basePromptFile",
                    help="Text file containing the base prompt to append the target paragraph to")
parser.add_argument("resultsDirectory", help="Directory to place the GPT-3 output JSON")
parser.add_argument("api_key", help="OpenAI API key")

args = parser.parse_args()

# So we can calculate the size of prompt and adjust max_token appropriately to keep under
# the hard limit.
# Note: GPT2Tokenizer doesn't match the GPT-3 Tokenizer 100%, but it is as good an estimate
#       as we can get. OpenAI suggested that we leverage it since they don't currently have a
#       truncate option in the API and the GPT-3 Tokenizer is not published.
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Max request size  (prompt itokens plus max returned tokens)
MAX_REQUEST_SIZE = 2049

# read in the command line args
keypart = args.keyRfilepartName
keyfile = args.keyfilePath + keypart
paraBasePath = args.paragraphDirectory
basePromptPath = args.basePromptFile
resultsFilePath = args.resultsDirectory + keypart + '.json'
apiKey = args.api_key

# Default params we will send to GPT-3 requests at OpenAi. They are described on
# the OpenAI documentation.
#
# Note:, we'll update the max_tokens on the fly if we think out base prompt and
# specific target paragraph will exceed the model length limit
params = {
        "max_tokens": 350,
        "temperature": 0.0,
        "engine": "davinci",
        "top_p": 1.0,
}

# Set your API key
openai.api_key = apiKey

# Read in the base prompt that we will append our target paragraphs to. Drop the
# newline characters to save on length
with open(basePromptPath, 'r') as basePromptFile:
    basePrompt = basePromptFile.read().strip('\n')

# Read in the keyfile to process. Keys are one per line and we will process each
# key in the file sequentially.
with open(keyfile, 'r') as keys:
    # Create file to save the JSON responses for all the keys in the keyfile
    with open(resultsFilePath, 'w') as resultFile:
        resultFile.write('{ "results": [')
        # Process a paragraph for a given key.  Query GPT-3
        for key in keys:
            # reset the max_tokens in case we overrode it on a previous call
            params["max_tokens"] = 350
            # start our output JSON for this paragraph
            resultFile.write('{ "doc": "' + key.strip('\n') + '",\n')

            # get the paragraph we want to submit for the given key
            evalParaPath = paraBasePath + key.strip('\n')
            with open(evalParaPath, 'r') as evalParaFile:
                evalPara = evalParaFile.read()

                # append the target paragraph to the base prompt.
                prmpt = basePrompt + "\n" + evalPara

                # try to estimate how many tokens that are in play and reduce the max_tokens if
                # we appended a long target paragraph that might exceed our limits. If it looks
                # it could be too long (which will return an error from GPT-3) we'll reduce the
                # number of tokens we ask to be returned to keep it under the limit.

                # GPT-2 tokenizer seems to generate less tokens than GPT-3 tokenizer in our tests.
                # We'll artificially bump the number of tokens to account for that.
                prmptTokCnt = len(tokenizer.encode(prmpt)) + 100
                if prmptTokCnt + 400 >= MAX_REQUEST_SIZE:
                    params["max_tokens"] = MAX_REQUEST_SIZE - prmptTokCnt

                # if we already know there is a length issue, don't bother submitting
                if params["max_tokens"] <= 1:
                    resultFile.write('"finish_reason":"token_length",\n')
                    resultFile.write('"text" : "''"\n')
                else:
                    # Send to GPT-3 and process the result
                    results = openai.Completion.create(prompt=prmpt, **params)
                    resultFile.write('"finish_reason":"' + results.choices[0].finish_reason +
                                     '",\n')
                    # Clean the returned results up so it is valid JSON when all is said and done.
                    resultFile.write('"text" : "' +
                                     results.choices[0].text.replace('\n', '\\n') + '"\n')
            resultFile.write('},\n')
        resultFile.write(']}')
