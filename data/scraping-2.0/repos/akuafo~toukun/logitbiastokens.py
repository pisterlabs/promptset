# Purpose of this test:  Using logitbias and tokens to force openai API to respond with strict output

# this code was inspoired by @AAAzzam:
# https://twitter.com/AAAzzam/status/1669753721574633473

import tiktoken
import time
import openai

def API_ChatCompletion(model, messages, logitbias):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1,
        logit_bias=logitbias
    )
    return response

encoding = tiktoken.get_encoding("cl100k_base")  # Using the cl100k_base model.

logitbias = {}




"""

From ChatGPT query...

A "logit" in this context is the raw output of a model's neural network for each possible next item, before it's turned into a probability distribution. These logits can be thought of as how strongly the model believes each potential output should be chosen.

"Sampling" refers to the process of choosing the actual output from these possibilities. After the logits are generated, they are converted to a probability distribution (using a softmax function), and the next item is sampled from this distribution.

The "logit_bias" parameter is used to add a bias or adjustment to these logits before the sampling process. This can be used to make certain outputs more or less likely. For instance, in a language model, you might add a logit bias to make some words more likely to be chosen as the next output.

For instance, if you're trying to bias the model to generate the phrase "artificial intelligence", which might be tokenized into two separate tokens "artificial" and "intelligence", applying bias to these two tokens independently won't necessarily make the model more likely to output them together as a phrase. It will only make each individual token more likely to appear, but not necessarily in that order or as a pair.

...

softmax function:  Amplifies the input values through an exponential function which makes the output probabilities more distinct and helps the model to be more decisive in its predictions.

This characteristic is particularly useful when you want a clear distinction between predictions. For instance, if you're classifying an image of a cat, you would prefer the model to output a high probability for the 'cat' class and very low probabilities for all other classes, rather than having similar probabilities for multiple classes.

In other words, the amplification property of the softmax function helps in producing a more definitive and interpretable prediction, which is often desirable in many machine learning tasks.

"""

print("\nIn this exercise, we will create a list of logitbias parameters with value 100 to force the OpenAI API to respond with a specific output.");time.sleep(1)
print("For a simple example, try 'true' and false' along with the test queries of 'does 2 + 2 = 4' and 'does 2 + 2 = 5'");time.sleep(1)
print("The challenge is encapsulating your intended logic in a single 'word' that's in the tiktoken dictionary.");time.sleep(1)
print("After you enter a word for logit bias, and we will try to encode it as a token.\n");time.sleep(1)
while True:
    bias_input = input("\nEnter your logitbias parameters (e.g. 'true' and 'false'), one at a time.  When you have entered them all, type 'done'. \n")
    if bias_input.lower() == 'done':
        print("\nYour logitbias tokens are:  " + str(logitbias) + "\n")
        break
    else:
        try:
            ## use tiktoken to get the token value
            token = encoding.encode(bias_input)[0]
            print("  Tiktoken byte string:  " + str(encoding.decode_bytes([token])) + " | UTF-8 decoded:  " + encoding.decode_bytes([token]).decode('utf-8', errors='replace'))
            use_token = input("  Confirm if this parameter is correct ('y', 'n')?  \n")
            if use_token.lower() == 'y':
                logitbias[token] = 100
        except ValueError:
            print("  Invalid input.")

print("\nlogitbias tokens:")
print(str(logitbias) + "\n")

model="gpt-4-0314"  #cl100k_base

messages = [
    {
        "role": "system",
        "content":  "You are a helpful assistant."
    }
]

while True:
    user_input = input("\nPlease enter a query for OpenAI (or 'q' for quit): \n")
    if user_input.lower() in ['quit', 'q']:
        break
    else:
        # Add the user's message to the list
        messages.append({
            "role": "user",
            "content": user_input
        })
        # Call the API function with the updated messages list
        response = API_ChatCompletion(model, messages, logitbias)
        if response['choices'][0]['message']['content']:
            print("\nThe OpenAI API response is:  " + response['choices'][0]['message']['content'])
        else:
            print("\nNo message content received.  The entire OpenAI API response is:  \n\n" + str(response) + "\n")
