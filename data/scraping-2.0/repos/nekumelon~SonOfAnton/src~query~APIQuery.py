"""
APIQuery should start by checking if a cached completion exists. If it doesn't-
prompt the API for a completion, cache it and use it.
"""

import openai, os, json, re, IO.io as io
from encoder.encoder import get_encoder

MAX_TOKENS = 2048;

with open('config.json') as configFile:
    config = json.loads(configFile.read());

openai.api_key = config['OPENAI_API_KEY'];
                
encoder = get_encoder();

def clamp(num, min_value, max_value): # https://www.tutorialspoint.com/How-to-clamp-floating-numbers-in-Pythons
   return max(min(num, max_value), min_value);

def tokenizePrompt(prompt):
    return encoder.encode(prompt);

def handleCompletion(completion): # Get the text of a completion and prepare it
    assert completion and completion != '', 'Unable to handle no/blank completion';

    completionText = completion['choices'][0]['text']; # Get the text from the first (Best) completion choice
    completionText = re.sub(r'^\s+', '', completionText); # Remove the new lines from the start of the text

    return completionText;

def getCachedCompletions():
    if (os.path.exists('completionsCache.json')):
        with open('completionsCache.json', 'r') as cacheFile:  
            try:
                completions = json.loads(cacheFile.read());

                return completions;
            except (json.JSONDecodeError):
                return {};

    return {};

def getCachedCompletion(prompt):
    completions = getCachedCompletions();
    
    if (prompt in completions):
        return completions[prompt];

def cacheCompletion(prompt, completion):
    completions = getCachedCompletions();
    completions[prompt] = completion;

    with open('completionsCache.json', 'w') as cacheFile:
        cacheFile.write(json.dumps(completions));

def promptGPT3(prompt, APIEngine, maxTokens):
    cachedCompletion = getCachedCompletion(prompt);

    if (cachedCompletion):
        io.out(handleCompletion(cachedCompletion));

        return;

    tokens = tokenizePrompt(prompt);

    completion = openai.Completion.create(
        engine = APIEngine,
        prompt = prompt,
        temperature = 0.65,
        max_tokens = clamp(MAX_TOKENS - len(tokens), 1, maxTokens)
    );
    
    if (completion and 'choices' in completion):
        cacheCompletion(prompt, completion);
        io.out(handleCompletion(completion));

        return;

    io.out('Sorry. I don\'t know that one.');

def APIQuery(query, APIEngine, maxTokens):
    promptGPT3(query, APIEngine, maxTokens);