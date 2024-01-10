from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List
import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

def ask_chatgpt(query: str) -> str:
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": query}])
    return completion.choices[0]['message']['content']

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:\n"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:\n"""

SUMMARY_PROMPT = """The input provides one or more comments from a social media discourse. Give a brief summary of the provided comments.
e.g. 
# Input: "@user1 @user2 @user3 I think you're all wrong. The best way to do this is to use a dictionary."
# Response: "The commentor thinks that others are wrong and that the best way to do this is to use a dictionary."

## Input: "@geeksquad"
## Response: "The comment tags another user."

#+# Input: "A quick brown fox jumps over the lazy dog."
#+# Response: "A fox jumps over a dog."

#+# Input: "Micheal"
#+# Response: "The commentor is likely referring to a person named Micheal."

#-# Input: ""
#-# Response: "The commentor is likely referring to a person named Micheal."

##+ Input: I love this new product! It's exactly what I've been looking for.
##+ Response: The commenter is happy with the product and would recommend it to others.

##- Input: This is the worst customer service I've ever experienced. I've been waiting for a response for weeks.
##- Response: The commenter is unhappy with the customer service they have received and is expressing their frustration.

##- Input: I'm not sure what I think of this. I'll have to try it out and see.
##- Response: The commenter is not sure how they feel about the product or service and is reserving judgment until they have more information.

## Input: Can you tell me more about this feature?
## Response: The commenter is asking for more information about the product or service.

# Input: I think this could be improved by adding a few more features.
# Response: The commenter is providing feedback on how the product or service could be improved.
"""

PARAPHRASE_PROMPT = """The input provides a comment from a social media discourse. Paraphrase the comment.
e.g.
##+ Input: Excited to see @TaylorSwift perform at the #BBMAs tonight! #SwiftiesUnite
##+ Response: I am excited to see Taylor Swift perform at the Billboard Music Awards tonight. I am using the hashtags #BBMAs and #SwiftiesUnite to show my support for Taylor Swift and her fans.

+## Input: Having the worst customer service experience with @Comcast. Been waiting for a response for weeks! #ComcastIsTheWorst
+## Response: I am having a negative customer service experience with Comcast. I have been waiting for a response for weeks. I am using the hashtag #ComcastIsTheWorst to express my frustration.

## Input: Not sure what I think of this new #iPhone. Will have to try it out and see. #AppleEvent
## Response: I am not sure how I feel about the new iPhone. I will have to try it out and see. I am using the hashtags #iPhone and #AppleEvent to show that I am interested in the product.

-## Input: Can you tell me more about this #feature? Not sure I understand.
-## Response: I am asking for more information about a specific feature. I am not sure I understand. I am using the hashtag #feature to make it clear what I am asking about.

#+# Input: @google @aoceu #Good
#+# Response: @google @aoceu #Good

#-# Input: This is amazing! So glad I found this #product. #LifeChanging
#-# Response: I am happy with a product or service that I have recently discovered. I am so glad I found this product. It is life-changing. I am using the hashtags #product and #LifeChanging to express my enthusiasm.

##- Input: So sorry to hear about what you're going through. #ThoughtsAndPrayers
##- Response: I am expressing my sympathy for someone who is going through a difficult time. I am so sorry to hear about what you are going through. I am thinking of you and sending you my prayers. I am using the hashtag #ThoughtsAndPrayers to show my support.

#-# Input: Micheal
#-# Response: Micheal

#+# Input: How do you print a string in Python? #Python
#+# Response: What is the method by which text can be printed in Python?.

# Input: This is hilarious! Sharing it with my friends. #FunnyMeme
# Response: I find a meme to be funny and am sharing it with my friends. This is hilarious! I am sharing it with my friends. I am using the hashtag #FunnyMeme to make it clear what I am sharing.
"""

SUMMARY_PROMPT_1 = """
TEXT:
```
@user1 @user2 @user3 I think you're all wrong. The best way to do this is to use a dictionary."
SUMMARY:
```
The commentor thinks that others are wrong and that the best way to do this is to use a dictionary.
```

TEXT:
```
@geeksquad"
SUMMARY:
```
The comment tags another user.
```

TEXT:
```
A quick brown fox jumps over the lazy dog.
```
SUMMARY:
```
A fox jumps over the dog.
```

TEXT:
```
Micheal
```
SUMMARY:
```
The commentor is likely referring to a person named Micheal.
```

TEXT:
```
I love this new product! It's exactly what I've been looking for.
```
SUMMARY:
```
The commenter is happy with the product and would recommend it to others.
```

TEXT:
```
This is the worst customer service I've ever experienced. I've been waiting for a response for weeks.
```
SUMMARY:
```
The commenter is unhappy with the customer service they have received and is expressing their frustration.
```

TEXT:
```
I'm not sure what I think of this. I'll have to try it out and see.
```
SUMMARY:
```
The commenter is not sure how they feel about the product or service and is reserving judgment until they have more information.
```

TEXT:
```
Can you tell me more about this feature?
```
SUMMARY:
```
The commenter is asking for more information about the product or service.
```

TEXT:
```
I think this could be improved by adding a few more features.
```
SUMMARY:
```
The commenter is providing feedback on how the product or service could be improved.
```

TEXT:
```
{text}
```
SUMMARY:
```
"""

PARAPHRASE_PROMPT_1 = """TEXT:
```
Excited to see @TaylorSwift perform at the #BBMAs tonight! #SwiftiesUnite
```
PARAPHRASE:
```
I am excited to see Taylor Swift perform at the Billboard Music Awards tonight. I am using the hashtags #BBMAs and #SwiftiesUnite to show my support for Taylor Swift and her fans.
```

TEXT:
```
Having the worst customer service experience with @Comcast. Been waiting for a response for weeks! #ComcastIsTheWorst
```
PARAPHRASE:
```
I am having a negative customer service experience with Comcast. I have been waiting for a response for weeks. I am using the hashtag #ComcastIsTheWorst to express my frustration.
```

TEXT:
```
Not sure what I think of this new #iPhone. Will have to try it out and see. #AppleEvent
```
PARAPHRASE:
```
I am not sure how I feel about the new iPhone. I will have to try it out and see. I am using the hashtags #iPhone and #AppleEvent to show that I am interested in the product.
```

TEXT:
```
Can you tell me more about this #feature? Not sure I understand.
```
PARAPHRASE:
```
I am asking for more information about a specific feature. I am not sure I understand. I am using the hashtag #feature to make it clear what I am asking about.
```

TEXT:
```
@google @aoceu #Good
```
PARAPHRASE:
```
@google @aoceu #Good
```

TEXT:
```
Micheal
```
PARAPHRASE:
```
Micheal
```

TEXT:
```
How do you print a string in Python? #Python
```
PARAPHRASE:
```
What is the method by which text can be printed in Python?.
```

TEXT:
```
This is hilarious! Sharing it with my friends. #FunnyMeme
```
PARAPHRASE:
```
I find a meme to be funny and am sharing it with my friends. This is hilarious! I am sharing it with my friends. I am using the hashtag #FunnyMeme to make it clear what I am sharing.
```

TEXT:
```
{text}
```
PARAPHRASE:
```
"""

TMP = """Miss Brill' is the story of an old woman told brilliantly and realistically, balancing thoughts and emotions that sustain her late solitary life amidst all the bustle of modern life. Miss Brill is a regular visitor on Sundays to the Jardins Publiques (the Public Gardens) of a small French suburb where she sits and watches all sorts of people come and go. She listens to the band playing, loves to watch people and guess what keeps them going, and enjoys contemplating the world as a great stage upon which actors perform. She finds herself to be another actor among the so many she sees, or at least herself as 'part of the performance after all.' One Sunday Miss Brill puts on her fur and goes to the Public Gardens as usual. The evening ends with her sudden realization that she is old and lonely, a realization brought to her by a conversation she overhears between a boy and a girl, presumably lovers, who comment on her unwelcome presence in their vicinity. Miss Brill is sad and depressed as she returns home, not stopping by as usual to buy her Sunday delicacy, a slice of honey-cake. She retires to her dark room, puts the fur back into the box and imagines that she has heard something cry."""
# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
# model = AutoModelForCausalLM.from_pretrained("/media/anique/Data/projects/llama-weights/llama2-7B-merged",
#                                                    load_in_8bit=True, device_map='auto')
# # use better transformer
# model.to_bettertransformer()
# model.tie_weights()
# # compile model
# model = torch.compile(model)
#
# tokenizer = AutoTokenizer.from_pretrained("/media/anique/Data/projects/llama-weights/llama2-7B", padding=True, padding_side='left')
# tokenizer.pad_token = tokenizer.eos_token
# {'shared': 0, 'decoder.embed_tokens': 0, 'encoder': 0, 'lm_head': 0, 'decoder.block.0': 0, 'decoder.block.1': 0, 'decoder.block.2': 0, 'decoder.block.3': 0, 'decoder.block.4': 0, 'decoder.block.5': 0, 'decoder.block.6': 'cpu', 'decoder.block.7': 'cpu', 'decoder.block.8': 'cpu', 'decoder.block.9': 'cpu', 'decoder.block.10': 'cpu', 'decoder.block.11': 'cpu', 'decoder.block.12': 'cpu', 'decoder.block.13': 'cpu', 'decoder.block.14': 'cpu', 'decoder.block.15': 'cpu', 'decoder.block.16': 'cpu', 'decoder.block.17': 'cpu', 'decoder.block.18': 'cpu', 'decoder.block.19': 'cpu', 'decoder.block.20': 'cpu', 'decoder.block.21': 'cpu', 'decoder.block.22': 'cpu', 'decoder.block.23': 'cpu', 'decoder.block.24': 'cpu', 'decoder.block.25': 'cpu', 'decoder.block.26': 'cpu', 'decoder.block.27': 'cpu', 'decoder.block.28': 'cpu', 'decoder.block.29': 'cpu', 'decoder.block.30': 'cpu', 'decoder.block.31': 'cpu', 'decoder.final_layer_norm': 'cpu', 'decoder.dropout': 'cpu'}

def extract_response(text, eos_token="</s>"):
    return text.split("### Response:")[1].split("#")[0].strip().replace(eos_token, "")

def extract_responses_summary(text):
    pre = SUMMARY_PROMPT_1.split('{text}')[0]
    return text.split(pre)[1].split('SUMMARY:\n```')[1].split('```')[0].strip().replace("</s>", "")

def extract_responses_paraphrase(text):
    pre = PARAPHRASE_PROMPT_1.split('{text}')[0]
    return text.split(pre)[1].split('PARAPHRASE:\n```')[1].split('```')[0].strip().replace("</s>", "")


def get_response(question):
    inputs = tokenizer(question, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(inputs, max_length=1024)
    return tokenizer.decode(outputs[0])

def get_response_batched(questions, model=None, tokenizer=None):
    inputs = tokenizer(questions, padding=True, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=1024)
    return [tokenizer.decode(o) for o in outputs]

def get_summary(text):
    prompt = generate_prompt(SUMMARY_PROMPT, text)
    return extract_response(get_response(prompt))


def get_paraphrase(text):
    prompt = generate_prompt(PARAPHRASE_PROMPT, text)
    return extract_response(get_response(prompt))

def get_summary_batched(texts: List[str]):
    prompts = [SUMMARY_PROMPT_1.format(text=text) for text in texts]
    responses = get_response_batched(prompts)
    return [extract_responses_summary(r) for r in responses]

def get_paraphrase_batched(texts: List[str], model=None, tokenizer=None):
    prompts = [PARAPHRASE_PROMPT_1.format(text=text) for text in texts]
    responses = get_response_batched(prompts, model=model, tokenizer=tokenizer)
    return [extract_responses_paraphrase(r) for r in responses]



# if __name__ == "__main__":
#     texts_to_summarize = [
#         "A quick brown fox jumped over the lazy dog.",
#         TMP,
#     ]
#     print(get_paraphrase_batched(texts_to_summarize))
